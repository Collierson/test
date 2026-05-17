"""
baseline_code_annotated.py

多任務 LSTM 模型：預測乒乓球比賽的下一個動作(actionId)、
得分類型(pointId)，以及發球方是否得分(serverGetPoint)。

【穩定優化版本】
- BiLSTM + 標準 CrossEntropy Loss（移除有問題的 Focal Loss）
- 簡化融合層，確保梯度流動正常
- 降低學習率以穩定訓練
- 保留核心改進：BiLSTM、更高權重、模型容量提升
"""

import argparse       # 解析命令列參數
import random         # 亂數生成（用於固定種子）
import numpy as np    # 數值運算
import pandas as pd   # 資料處理
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split   # 切分訓練/驗證集
from sklearn.metrics import f1_score, roc_auc_score    # 評估指標

# ─────────────────────────────────────────────────────────────
# 固定隨機種子，確保每次執行結果可重現
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 用於序列建模的特徵欄位（皆為類別型變數）
FEATURES = [
    "sex",          # 球員性別
    "handId",       # 慣用手
    "strengthId",   # 擊球力道
    "spinId",       # 旋轉類型
    "pointId",      # 得分類型
    "actionId",     # 擊球動作
    "positionId",   # 場上位置
    "strikeId",     # 擊球方式
    "scoreSelf",    # 自身得分
    "scoreOther",   # 對手得分
    "strikeNumber"  # 本回合擊球編號
]
PAD_TOKEN = 0  # 填充值（padding），用於對齊不同長度的序列


# ─────────────────────────────────────────────────────────────
# 資料集類別
# ─────────────────────────────────────────────────────────────
class RallyDataset(Dataset):
    """
    將每一個 rally（回合）包裝成 PyTorch Dataset。
    每筆資料包含：
      - X   : 輸入特徵序列 (MAXLEN, num_features)，long 整數
      - yA  : 下一拍動作標籤序列 (MAXLEN,)，long 整數
      - yP  : 下一拍得分類型標籤序列 (MAXLEN,)，long 整數
      - yR  : 發球方是否得分 (純量)，float
      - L   : 序列實際長度 (未填充的拍數)
    """
    def __init__(self, X, yA, yP, yR, L):
        # 轉換為 PyTorch tensor
        self.X  = torch.tensor(X,  dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)

    def __len__(self):
        # 回傳資料集中的回合數量
        return self.X.shape[0]

    def __getitem__(self, i):
        # 取得第 i 個回合的所有資料
        return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]


# ─────────────────────────────────────────────────────────────
# 改進的多任務 BiLSTM 模型
# ─────────────────────────────────────────────────────────────
class ImprovedMultiTaskLSTM(nn.Module):
    """
    【穩定改進版本】
    1. BiLSTM (雙向) - 利用前後文脈，性能更好
    2. 簡化架構 - 直接的輸出層，確保梯度流動
    3. 標準 CrossEntropy Loss - 移除 Focal Loss
    4. 增加模型容量 - 更深、更寬的網絡
    """
    def __init__(self, num_tokens_per_feature, n_act, n_pt,
                 emb_dim=20, hidden=160, num_layers=2, dropout=0.25):
        super().__init__()

        # 為每個類別特徵建立獨立的 Embedding 層
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_tokens_per_feature
        ])

        # 雙向 LSTM：輸入維度 = 特徵數 × embedding 維度
        self.lstm = nn.LSTM(
            len(num_tokens_per_feature) * emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True  # ← BiLSTM: 輸出維度會是 hidden*2
        )

        self.drop = nn.Dropout(dropout)

        # BiLSTM 的輸出維度是 hidden*2（前向 + 反向）
        lstm_output_dim = hidden * 2

        # 三個任務的輸出頭（直接線性層，無融合層以避免問題）
        self.act_head = nn.Linear(lstm_output_dim, n_act)   # 動作分類
        self.pt_head  = nn.Linear(lstm_output_dim, n_pt)    # 得分類型分類
        self.rly_head = nn.Linear(lstm_output_dim, 1)       # 回合結果（二元分類）

    def forward(self, X, lengths):
        """
        X       : (batch, MAXLEN, num_features) 輸入特徵
        lengths : (batch,) 每個回合的實際長度
        回傳：
          la  : (batch, MAXLEN, n_act)  — 動作 logits
          lp  : (batch, MAXLEN, n_pt)   — 得分 logits
          lr  : (batch,)                — 回合結果 logit（未過 sigmoid）
        """
        # 每個特徵分別做 embedding，然後在最後一維拼接
        es = [emb(X[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)  # (batch, MAXLEN, num_features * emb_dim)

        # 打包序列以忽略 padding，提升計算效率
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        o, _ = self.lstm(packed)
        # 解包回原始形狀
        o, _ = nn.utils.rnn.pad_packed_sequence(
            o, batch_first=True, total_length=X.size(1)
        )
        o = self.drop(o)  # dropout

        # 建立遮罩：只有非 PAD 的位置為 1
        mask  = (X[:, :, 0] != PAD_TOKEN).float().unsqueeze(-1)  # (batch, MAXLEN, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)  # 避免除以 0

        # 對所有非 PAD 時間步取平均，作為回合整體表示
        mean_hidden = (o * mask).sum(dim=1) / denom  # (batch, lstm_output_dim)

        # 回傳三個任務的輸出
        return (
            self.act_head(o),                        # 逐步動作預測
            self.pt_head(o),                         # 逐步得分預測
            self.rly_head(mean_hidden).squeeze(1)    # 回合結果預測
        )


# ─────────────────────────────────────────────────────────────
# 工具函數：填充序列至固定長度
# ─────────────────────────────────────────────────────────────
def pad2d(a, m, pad_val=PAD_TOKEN):
    """
    將 2D 陣列 a（形狀 T × F）填充至長度 m。
    超出部分填入 pad_val（預設 0）。
    """
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out


def pad1d(a, m, ignore_index=-1):
    """
    將 1D 標籤陣列填充至長度 m。
    填充值為 -1（CrossEntropyLoss 的 ignore_index，訓練時忽略）。
    """
    out = np.full((m,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out


# ─────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────
def main(args):
    # ── 1. 讀取資料 ──────────────────────────────────────────
    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])
    sub   = pd.read_csv(args.sample)  # 提交範本（只取格式）

    # 截斷異常大的 strikeNumber（最多 40 拍）
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    # ── 2. 類別編碼 ──────────────────────────────────────────
    # 用訓練集的類別集合建立編碼字典（避免測試集引入未知類別）
    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}

    def encode_frame(df):
        """將 DataFrame 轉為整數編碼矩陣（形狀 T × num_features）。
        Categorical codes 從 -1 開始（未知類別），+1 後 0 保留給 PAD_TOKEN。
        """
        outs = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)  # (T, num_features)

    # ── 3. 建立訓練樣本（以回合為單位）──────────────────────
    X_list, yA_list, yP_list, yR_list, L_list = [], [], [], [], []

    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2:
            continue  # 至少需要 2 拍才能做「下一拍預測」

        # 輸入：第 1 到倒數第 2 拍（去掉最後一拍）
        X  = encode_frame(g)[:-1]
        # 目標：第 2 到最後一拍（每個位置預測「下一拍」）
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)

        X_list.append(X)
        yA_list.append(yA)
        yP_list.append(yP)
        yR_list.append(int(g["serverGetPoint"].iloc[0]))  # 回合結果（固定值）
        L_list.append(len(X))  # 序列實際長度

    # ── 4. 填充至最大長度，堆疊成矩陣 ───────────────────────
    MAXLEN = max(L_list)
    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])   # (N, MAXLEN, F)
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])  # (N, MAXLEN)
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])  # (N, MAXLEN)
    yR_all = np.array(yR_list, dtype=np.float32)             # (N,)
    L_all  = np.array(L_list,  dtype=np.int64)               # (N,)

    # ── 5. 重新映射標籤至連續整數（CrossEntropyLoss 需要） ──
    # 動作類別
    act_classes = np.sort(train["actionId"].unique())
    n_act       = len(act_classes)
    act_id2idx  = {v: i for i, v in enumerate(act_classes)}

    # 得分類型類別
    pt_classes  = np.sort(train["pointId"].unique())
    n_pt        = len(pt_classes)
    pt_id2idx   = {v: i for i, v in enumerate(pt_classes)}

    # 將原始 ID 映射為連續索引（未知值保持 -1，訓練時忽略）
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
    yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)

    # ── 6. 切分訓練集與驗證集（依回合結果做分層抽樣）────────
    idx = np.arange(len(X_all))
    tr_idx, va_idx = train_test_split(
        idx, test_size=args.val_size, random_state=42,
        stratify=(yR_all > 0.5)  # 保持正負比例
    )
    X_tr,  X_va  = X_all[tr_idx],  X_all[va_idx]
    yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]
    yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
    yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
    L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]

    # ── 7. 計算類別權重（解決類別不平衡問題）────────────────
    # 頻率越低的類別，權重越高
    act_counts = np.bincount(yA_tr[yA_tr != -1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr != -1].ravel(), minlength=n_pt)  + 1

    act_w = torch.tensor(1.0 / act_counts, dtype=torch.float32)
    act_w = act_w * (n_act / act_w.sum())   # 正規化使總和 = n_act

    pt_w  = torch.tensor(1.0 / pt_counts,  dtype=torch.float32)
    pt_w  = pt_w  * (n_pt  / pt_w.sum())

    # ── 8. DataLoader ────────────────────────────────────────
    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=max(args.batch * 2, 128), shuffle=False)

    # ── 9. 建立模型與損失函數 ─────────────────────────────────
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]  # 每個特徵的 token 數
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedMultiTaskLSTM(
        num_tokens_per_feature, n_act, n_pt,
        emb_dim=args.emb, hidden=args.hidden,
        num_layers=args.layers, dropout=args.drop
    ).to(device)

    # 標準加權交叉熵（更穩定，移除 Focal Loss）
    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    # 二元交叉熵（回合結果為 0/1）
    bce_rally = nn.BCEWithLogitsLoss()

    # 降低學習率以提升穩定性
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_score = -1
    wait = 0
    patience = 12

    # ── 10. 訓練迴圈 ──────────────────────────────────────────
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0

        for Xb, yAb, yPb, yRb, Lb in train_loader:
            Xb, yAb, yPb, yRb, Lb = (
                Xb.to(device), yAb.to(device),
                yPb.to(device), yRb.to(device), Lb.to(device)
            )
            opt.zero_grad()

            # 前向傳播
            la, lp, lr = model(Xb, Lb)

            # 多任務損失：提高 actionId 權重 (50% + 35% + 15%)
            loss = (
                0.50 * ce_action(la.view(-1, la.size(-1)), yAb.view(-1)) +
                0.35 * ce_point(lp.view(-1, lp.size(-1)),  yPb.view(-1)) +
                0.15 * bce_rally(lr, yRb)
            )

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * Xb.size(0)

        # ── 11. 驗證 ─────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        allA, allAp = [], []   # 動作：真實值 & 預測值
        allP, allPp = [], []   # 得分：真實值 & 預測值
        allR, allRp = [], []   # 回合：真實值 & 預測機率

        with torch.no_grad():
            for Xb, yAb, yPb, yRb, Lb in val_loader:
                Xb, yAb, yPb, yRb, Lb = (
                    Xb.to(device), yAb.to(device),
                    yPb.to(device), yRb.to(device), Lb.to(device)
                )
                la, lp, lr = model(Xb, Lb)

                loss = (
                    0.50 * ce_action(la.view(-1, la.size(-1)), yAb.view(-1)) +
                    0.35 * ce_point(lp.view(-1, lp.size(-1)),  yPb.view(-1)) +
                    0.15 * bce_rally(lr, yRb)
                )
                val_loss += loss.item() * Xb.size(0)

                # 收集回合結果預測
                allR  += yRb.detach().cpu().tolist()
                allRp += torch.sigmoid(lr).detach().cpu().tolist()

                # 收集動作與得分的預測
                yA_flat = yAb.view(-1).detach().cpu().numpy()
                yP_flat = yPb.view(-1).detach().cpu().numpy()
                a_pred  = la.argmax(-1).view(-1).detach().cpu().numpy()
                p_pred  = lp.argmax(-1).view(-1).detach().cpu().numpy()

                mA = (yA_flat != -1)  # 過濾填充位置
                mP = (yP_flat != -1)
                allA  += yA_flat[mA].tolist()
                allAp += a_pred[mA].tolist()
                allP  += yP_flat[mP].tolist()
                allPp += p_pred[mP].tolist()

        # ── 12. 計算評估指標 ──────────────────────────────────
        tr_loss = run_loss / len(train_loader.dataset)
        va_loss = val_loss / len(val_loader.dataset)

        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1A, f1P, auc = 0.0, 0.0, 0.5

        # 最終加權分數
        final = 0.50 * f1A + 0.35 * f1P + 0.15 * auc
        print(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} "
            f"Final~{final:.4f}"
        )

        # 學習率調度
        scheduler.step(final)

        # 提前結束訓練
        if final > best_score:
            best_score = final
            wait = 0
            print("已保存最佳模型")
            torch.save(model.state_dict(), f"best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break


    # ─────────────────────────────────────────────────────────
    # 13. 推論（對測試集生成預測）
    # ─────────────────────────────────────────────────────────
    def pad2d_cap(a, m, pad_val=PAD_TOKEN):
        """
        與 pad2d 相同，但截斷超過 m 的序列（避免 MAXLEN 溢出）。
        回傳 (填充後陣列, 實際有效長度 T)。
        """
        out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
        T = min(len(a), m)
        out[:T] = a[:T]
        return out, T

    # 載入最佳模型
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    pred_rows = []
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            Xg = encode_frame(g)          # 編碼測試回合
            Xp, T = pad2d_cap(Xg, MAXLEN)

            X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)],   dtype=torch.long, device=device)

            la, lp, lr = model(X_t, L_t)

            # 取最後一個有效時間步的預測
            last_t  = L_t.item() - 1
            a_idx   = int(torch.argmax(la[0, last_t]).item())
            p_idx   = int(torch.argmax(lp[0, last_t]).item())
            s_prob  = float(torch.sigmoid(lr).item())

            # 將索引轉回原始類別 ID
            action_pred = int(act_classes[a_idx])
            point_pred  = int(pt_classes[p_idx])

            pred_rows.append({
                "rally_uid":      int(rid),
                "actionId":       action_pred,
                "pointId":        point_pred,
                "serverGetPoint": s_prob,
            })

    # ─────────────────────────────────────────────────────────
    # 14. 輸出提交檔案
    # ─────────────────────────────────────────────────────────
    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")

    # 與提交範本合併（按 rally_uid 對齊），覆蓋預測欄位
    out = pred_df[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out.to_csv(args.out, index=False)
    print(f"Saved submission to: {args.out}")
    print(out.head())


# ─────────────────────────────────────────────────────────────
# 命令列介面
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",    default="train.csv")            # 訓練資料路徑
    ap.add_argument("--test",     default="old_test.csv")         # 測試資料路徑
    ap.add_argument("--sample",   default="sample_submission.csv")# 提交範本路徑
    ap.add_argument("--out",      default="submission_lstm_baseline.csv") # 輸出路徑
    ap.add_argument("--epochs",   type=int,   default=25)         # 訓練 epoch 數
    ap.add_argument("--batch",    type=int,   default=64)        # 批次大小
    ap.add_argument("--emb",      type=int,   default=20)        # Embedding 維度
    ap.add_argument("--hidden",   type=int,   default=160)       # LSTM 隱藏層維度
    ap.add_argument("--layers",   type=int,   default=2)         # LSTM 層數
    ap.add_argument("--drop",     type=float, default=0.25)      # Dropout 比例
    ap.add_argument("--lr",       type=float, default=5e-4)      # 學習率（降低）
    ap.add_argument("--val_size", type=float, default=0.10)      # 驗證集比例
    args = ap.parse_args()
    main(args)
