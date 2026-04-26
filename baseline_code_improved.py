"""
乒乓球戰術預測 v3-fix — 修正 target 索引 + 梯度爆炸問題
=============================================================
【根本原因修正】
✅ FocalLoss 的 target 範圍檢查排除 ignore_index (-1)，不再誤 return 0
✅ pt_map 納入 pointId=0（網/出界），讓 yP 有足夠有效樣本
✅ act_map 也納入 actionId=0，避免相同問題
✅ 梯度裁剪閾值從 1.0 改為 5.0，避免正常梯度也被過度壓制
✅ 移除會誤觸發的 target 範圍警告（改成靜默 clamp）

【功能保留】
✅ 語義特徵工程（九宮格座標/旋轉向量/戰術情境/球種類型/序列位置）
✅ 代號語義對應 action_type
✅ 驗證集 Early Stopping（基於 Overall Score）
✅ serverGetPoint 輸出機率值
✅ 損失權重 0.4/0.4/0.2 對齊評分公式
✅ WeightedRandomSampler
"""

import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score
import time
from datetime import timedelta
import os
import warnings

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 代號語義映射
# ══════════════════════════════════════════════════════════════════════════════
ACTION_TYPE_MAP = {
    0: 4,    # 無/其他 → 獨立類別（不用 -1，避免被 ignore）
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,   # Attack
    8: 2, 9: 2, 10: 2, 11: 2,                     # Control
    12: 3, 13: 3, 14: 3,                           # Defensive
    15: 0, 16: 0, 17: 0, 18: 0,                   # Serve
}

POINT_COORD_MAP = {
    0: (1, 1),   # 網/出界 → 對應中心座標（而非 -1，避免回歸損失異常）
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}

SPIN_VEC_MAP = {
    0: (0.0, 0.0), 1: (1.0, 0.0), 2: (-1.0, 0.0),
    3: (0.0, 0.0), 4: (0.5, 1.0), 5: (-0.5, 1.0),
}

CAT_FEATURES = ["sex", "handId", "strengthId", "spinId",
                "pointId", "actionId", "positionId", "strikeId"]
NUM_FEATURES  = ["scoreSelf", "scoreOther", "strikeNumber", "scoreDiff"]
PAD_TOKEN     = 0
IGNORE_IDX    = -1


# ══════════════════════════════════════════════════════════════════════════════
# 特徵工程
# ══════════════════════════════════════════════════════════════════════════════
def add_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["point_row"] = df["pointId"].map(lambda x: POINT_COORD_MAP.get(int(x), (1, 1))[0])
    df["point_col"] = df["pointId"].map(lambda x: POINT_COORD_MAP.get(int(x), (1, 1))[1])
    df["spin_v"]    = df["spinId"].map(lambda x: SPIN_VEC_MAP.get(int(x), (0.0, 0.0))[0])
    df["spin_h"]    = df["spinId"].map(lambda x: SPIN_VEC_MAP.get(int(x), (0.0, 0.0))[1])
    df["is_serve"]   = (df["strikeId"] == 1).astype(float)
    df["is_receive"] = (df["strikeId"] == 2).astype(float)
    df["tactical_ctx"] = (
        (df["handId"].clip(0, 2) - 1) * 3 +
        (df["positionId"].clip(0, 3) - 1)
    ).clip(0, 5) / 5.0
    act_type = df["actionId"].map(lambda x: ACTION_TYPE_MAP.get(int(x), 4))
    df["action_is_serve"]     = (act_type == 0).astype(float)
    df["action_is_attack"]    = (act_type == 1).astype(float)
    df["action_is_control"]   = (act_type == 2).astype(float)
    df["action_is_defensive"] = (act_type == 3).astype(float)
    df["strike_phase"] = 0.0
    return df


def compute_strike_phase(seq_df):
    n = len(seq_df)
    return np.zeros(n) if n <= 1 else np.arange(n) / (n - 1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class RallyDataset(Dataset):
    def __init__(self, X_cat, X_num, X_sem, yA, yP, yR, L):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_sem = torch.tensor(X_sem, dtype=torch.float32)
        self.yA    = torch.tensor(yA,    dtype=torch.long)
        self.yP    = torch.tensor(yP,    dtype=torch.long)
        self.yR    = torch.tensor(yR,    dtype=torch.float32)
        self.L     = torch.tensor(L,     dtype=torch.long)

    def __len__(self): return self.X_cat.shape[0]

    def __getitem__(self, i):
        return (self.X_cat[i], self.X_num[i], self.X_sem[i],
                self.yA[i], self.yP[i], self.yR[i], self.L[i])


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss — 正確處理 ignore_index
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    修正版 Focal Loss：
    - ignore_index (-1) 只在計算時排除，不影響 target 範圍檢查
    - 使用 log_softmax + nll_loss 保持數值穩定
    - 不做 target 範圍 early-return（讓 nll_loss 自己處理 ignore_index）
    """
    def __init__(self, alpha=0.75, gamma=2.0, weight=None, ignore_index=IGNORE_IDX):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.weight       = weight

    def forward(self, pred, target):
        # 先把 ignore_index 位置的 target 臨時替換成 0（讓 gather 不出錯）
        # 最後再把這些位置的 loss mask 掉
        valid_mask  = (target != self.ignore_index)
        target_safe = target.clone()
        target_safe[~valid_mask] = 0   # 填任意合法 index，後面 mask 掉

        log_p = F.log_softmax(pred, dim=-1)          # [N, C]
        ce    = F.nll_loss(
            log_p, target_safe,
            weight=self.weight,
            reduction='none',
            ignore_index=-100   # 已手動 mask，不靠 nll_loss 的 ignore
        )                                             # [N]

        p     = torch.exp(-ce)
        focal = self.alpha * (1.0 - p + 1e-7) ** self.gamma * ce

        # 只保留有效位置
        focal = focal * valid_mask.float()

        if valid_mask.any():
            return focal.sum() / valid_mask.sum().clamp(min=1)
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
class SemanticMultiTaskLSTM(nn.Module):
    def __init__(self, cat_dims, num_dim, sem_dim, n_act, n_pt,
                 emb_dim=32, hidden=256, num_layers=3, dropout=0.3):
        super().__init__()

        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in cat_dims
        ])

        lstm_in = len(cat_dims) * emb_dim + num_dim + sem_dim
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(hidden * 2)

        # ActionId 預測頭
        self.act_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.LayerNorm(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),     nn.LayerNorm(hidden),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_act)
        )

        # PointId 預測頭
        self.pt_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.LayerNorm(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),     nn.LayerNorm(hidden),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_pt)
        )

        # ServerGetPoint 預測頭（使用 mean+max 全序列池化）
        self.rly_pool_proj = nn.Linear(hidden * 4, hidden * 2)
        self.rly_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, X_cat, X_num, X_sem, lengths):
        B, T = X_cat.shape[:2]

        embs = [emb(X_cat[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(embs + [X_num, X_sem], dim=-1)
        x = self.input_proj(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=T)
        lstm_out = self.lstm_norm(lstm_out)

        # 最後一個有效步驟
        last_idx  = (lengths - 1).long()
        last_step = lstm_out[torch.arange(B), last_idx]

        # ActionId / PointId：用最後一步
        act_out = self.act_head(last_step)
        pt_out  = self.pt_head(last_step)

        # ServerGetPoint：mean+max 全序列池化
        pad_mask  = (X_cat[:, :, 0] == PAD_TOKEN)
        mask_f    = (~pad_mask).float().unsqueeze(-1)
        seq_mean  = (lstm_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        seq_max   = lstm_out.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(1).values
        pool_feat = self.rly_pool_proj(torch.cat([seq_mean, seq_max], dim=-1))
        rly_out   = self.rly_head(pool_feat).squeeze(-1)

        return act_out, pt_out, rly_out


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════
def format_time(s): return str(timedelta(seconds=int(s)))


def save_best_model(model, optimizer, epoch, score, path, device_name):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': score, 'device': device_name}, path)
    print(f"💾 最佳模型已儲存 (Score={score:.4f})")


def load_best_model(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"📂 載入最佳模型 Epoch={ckpt['epoch']} Score={ckpt['best_score']:.4f}")
    return model, optimizer, ckpt['epoch'], ckpt['best_score']


def get_loss_weights(labels, n_classes):
    """計算類別平衡權重，labels 中的 IGNORE_IDX 會被排除"""
    valid = labels[labels != IGNORE_IDX]
    if len(valid) == 0:
        return torch.ones(n_classes)
    present = np.unique(valid)
    raw_w   = compute_class_weight('balanced', classes=present, y=valid)
    full_w  = np.ones(n_classes)
    for idx, l in enumerate(present):
        if 0 <= l < n_classes:
            full_w[l] = raw_w[idx]
    return torch.tensor(full_w, dtype=torch.float32)


def compute_sample_weights(labels_2d, n_classes):
    valid_mask = (labels_2d != IGNORE_IDX)
    all_valid  = labels_2d[valid_mask]
    if len(all_valid) == 0:
        return np.ones(len(labels_2d))
    present = np.unique(all_valid)
    raw_w   = compute_class_weight('balanced', classes=present, y=all_valid)
    full_w  = np.ones(n_classes)
    for idx, l in enumerate(present):
        if 0 <= l < n_classes:
            full_w[l] = raw_w[idx]
    sample_w = []
    for row in labels_2d:
        valid_row = row[(row != IGNORE_IDX) & (row >= 0) & (row < n_classes)]
        w = np.mean([full_w[l] for l in valid_row]) if len(valid_row) > 0 else 1.0
        sample_w.append(w)
    return np.array(sample_w)


# ══════════════════════════════════════════════════════════════════════════════
# 評估（競賽評分公式）
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, device, act_classes, pt_classes):
    model.eval()
    all_ya, all_pa = [], []
    all_yp, all_pp = [], []
    all_yr, all_pr = [], []

    with torch.no_grad():
        for xc, xn, xs, ya, yp, yr, l in loader:
            xc = xc.to(device); xn = xn.to(device); xs = xs.to(device)
            ya = ya.to(device); yp = yp.to(device)
            yr = yr.to(device).float(); l = l.to(device)

            pa, pp, pr = model(xc, xn, xs, l)
            B    = xc.shape[0]
            last = (l - 1).long()

            ya_last = ya[torch.arange(B), last]
            yp_last = yp[torch.arange(B), last]

            # 只收集非 ignore 的樣本
            act_valid = ya_last != IGNORE_IDX
            pt_valid  = yp_last != IGNORE_IDX

            all_ya.extend(ya_last[act_valid].cpu().numpy())
            all_pa.extend(pa[act_valid].argmax(-1).cpu().numpy())
            all_yp.extend(yp_last[pt_valid].cpu().numpy())
            all_pp.extend(pp[pt_valid].argmax(-1).cpu().numpy())
            all_yr.extend(yr.cpu().numpy())
            all_pr.extend(torch.sigmoid(pr).cpu().numpy())

    f1_act = f1_score(all_ya, all_pa, average='macro', zero_division=0) if all_ya else 0.0
    f1_pt  = f1_score(all_yp, all_pp, average='macro', zero_division=0) if all_yp else 0.0
    try:
        auc = roc_auc_score(all_yr, all_pr) if len(set(all_yr)) >= 2 else 0.5
    except ValueError:
        auc = 0.5

    return 0.4 * f1_act + 0.4 * f1_pt + 0.2 * auc, f1_act, f1_pt, auc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    print("=" * 80)
    print("乒乓球戰術預測 v3-fix — 修正 target 索引 + 梯度爆炸")
    print("=" * 80)

    train_df = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test_df  = pd.read_csv(args.test) .sort_values(["rally_uid", "strikeNumber"])

    train_df["scoreDiff"] = train_df["scoreSelf"] - train_df["scoreOther"]
    test_df ["scoreDiff"] = test_df ["scoreSelf"] - test_df ["scoreOther"]

    train_df = add_semantic_features(train_df)
    test_df  = add_semantic_features(test_df)

    # 標準化（含語義連續特徵）
    num_cols = NUM_FEATURES + ["point_row", "point_col", "spin_v", "spin_h", "tactical_ctx"]
    for col in num_cols:
        m, s = train_df[col].mean(), train_df[col].std()
        train_df[col] = (train_df[col] - m) / (s + 1e-7)
        test_df [col] = (test_df [col] - m) / (s + 1e-7)

    cats = {c: pd.Categorical(train_df[c]).categories for c in CAT_FEATURES}

    def encode_cat(df):
        return np.stack(
            [pd.Categorical(df[c], categories=cats[c]).codes + 1 for c in CAT_FEATURES], axis=1
        )

    def encode_num(df):
        return np.stack([df[c].values for c in NUM_FEATURES], axis=1)

    def encode_sem(df, phase_arr=None):
        base_cols = ["point_row", "point_col", "spin_v", "spin_h",
                     "is_serve", "is_receive", "tactical_ctx",
                     "action_is_attack", "action_is_control",
                     "action_is_defensive", "action_is_serve"]
        data = np.stack([df[c].values.astype(np.float32) for c in base_cols], axis=1)
        ph = phase_arr if phase_arr is not None else np.zeros((len(df), 1))
        if ph.ndim == 1: ph = ph.reshape(-1, 1)
        return np.concatenate([data, ph], axis=1).astype(np.float32)

    # ── 序列構建 ──────────────────────────────────────────────────────────────
    X_cat_list, X_num_list, X_sem_list = [], [], []
    yA_list, yP_list, yR_list, L_list  = [], [], [], []

    for rid, g in train_df.groupby("rally_uid"):
        if len(g) < 2:
            continue
        phase = compute_strike_phase(g).reshape(-1, 1)
        xc = encode_cat(g); xn = encode_num(g); xs = encode_sem(g, phase)
        X_cat_list.append(xc[:-1]); X_num_list.append(xn[:-1]); X_sem_list.append(xs[:-1])
        yA_list.append(g["actionId"].values[1:])
        yP_list.append(g["pointId"].values[1:])
        yR_list.append(int(g["serverGetPoint"].iloc[0]))
        L_list.append(len(g) - 1)

    MAXLEN  = max(L_list)
    SEM_DIM = X_sem_list[0].shape[1]

    def pad2d(lst, pv=0):
        return np.array([np.pad(a, ((0, MAXLEN-len(a)), (0,0)), constant_values=pv) for a in lst])
    def pad1d(lst, pv=IGNORE_IDX):
        return np.array([np.pad(a, (0, MAXLEN-len(a)), constant_values=pv) for a in lst])

    X_cat_all = pad2d(X_cat_list)
    X_num_all = pad2d(X_num_list)
    X_sem_all = pad2d(X_sem_list, pv=0.0)
    yA_all    = pad1d(yA_list)
    yP_all    = pad1d(yP_list)
    yR_all    = np.array(yR_list)
    L_all     = np.array(L_list)

    # ── 類別映射 ──────────────────────────────────────────────────────────────
    # ★ 關鍵修正：pt_classes 和 act_classes 都包含 0
    #   0 代表「無落點/無動作」，是合法的預測目標（模型需要能預測「這拍不落台」）
    act_classes = np.sort(train_df["actionId"].unique())   # 含 0
    pt_classes  = np.sort(train_df["pointId"].unique())    # 含 0

    act_map = {v: i for i, v in enumerate(act_classes)}
    pt_map  = {v: i for i, v in enumerate(pt_classes)}

    yA_flat = np.vectorize(lambda x: act_map.get(int(x), IGNORE_IDX))(yA_all)
    yP_flat = np.vectorize(lambda x: pt_map.get(int(x), IGNORE_IDX))(yP_all)

    # 診斷：最後一拍的有效標籤數
    last_ya = yA_flat[np.arange(len(L_all)), L_all - 1]
    last_yp = yP_flat[np.arange(len(L_all)), L_all - 1]
    print(f"\n📊 資料統計：")
    print(f"  • 訓練樣本: {len(X_cat_all)} | 最大序列長度: {MAXLEN}")
    print(f"  • Action: {len(act_map)} 類（含 0={act_map.get(0,'?')}）")
    print(f"  • Point:  {len(pt_map)} 類（含 0={pt_map.get(0,'?')}）")
    print(f"  • 最後一拍 ActionId 有效標籤: {(last_ya>=0).sum()}/{len(L_all)}")
    print(f"  • 最後一拍 PointId  有效標籤: {(last_yp>=0).sum()}/{len(L_all)}")
    print(f"  • 語義特徵維度: {SEM_DIM}")

    # ── Loss 權重 ─────────────────────────────────────────────────────────────
    act_weights = get_loss_weights(yA_flat, len(act_map))
    pt_weights  = get_loss_weights(yP_flat, len(pt_map))

    # ── Train / Val 分割 ──────────────────────────────────────────────────────
    tr_idx, va_idx = train_test_split(
        np.arange(len(L_all)), test_size=0.1, stratify=yR_all, random_state=SEED
    )

    action_weights = compute_sample_weights(yA_flat[tr_idx], len(act_map))
    sampler = WeightedRandomSampler(action_weights, len(action_weights), replacement=True)

    def make_ds(idx):
        return RallyDataset(
            X_cat_all[idx], X_num_all[idx], X_sem_all[idx],
            yA_flat[idx], yP_flat[idx], yR_all[idx], L_all[idx]
        )

    train_loader = DataLoader(make_ds(tr_idx), batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(make_ds(va_idx), batch_size=args.batch, shuffle=False,  num_workers=0)

    # ── 模型 ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  設備: {device}")

    model = SemanticMultiTaskLSTM(
        cat_dims=[len(cats[c]) for c in CAT_FEATURES],
        num_dim=len(NUM_FEATURES),
        sem_dim=SEM_DIM,
        n_act=len(act_map),
        n_pt=len(pt_map),
    ).to(device)

    print(f"  • 參數數: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=5e-6
    )

    # ★ 使用修正版 FocalLoss（不做有問題的 target 範圍 early-return）
    loss_fn_act = FocalLoss(alpha=0.75, gamma=2.0, weight=act_weights.to(device))
    loss_fn_pt  = FocalLoss(alpha=0.75, gamma=2.0, weight=pt_weights.to(device))
    loss_fn_bin = nn.BCEWithLogitsLoss()

    print("\n" + "=" * 80)
    print("開始訓練")
    print("=" * 80)
    print(f"配置: Batch={args.batch}, LR={args.lr}, Epochs={args.epochs}")
    print("修正:")
    print("  ✅ FocalLoss：ignore_index 用 mask 而非 early-return 處理")
    print("  ✅ pt_classes / act_classes 含 0（避免 yP 全為 ignore）")
    print("  ✅ 梯度裁剪閾值 5.0（避免過度壓制正常梯度）")
    print("  ✅ 驗證集 Early Stopping（Overall Score）\n")

    best_score = -1.0
    patience = 20
    no_improve = 0
    start = time.time()

    for ep in range(1, args.epochs + 1):
        ep_start = time.time()
        model.train()
        losses = {"total": 0, "act": 0, "pt": 0, "rly": 0}
        nb = 0

        for xc, xn, xs, ya, yp, yr, l in train_loader:
            xc = xc.to(device); xn = xn.to(device); xs = xs.to(device)
            ya = ya.to(device); yp = yp.to(device)
            yr = yr.to(device).float(); l = l.to(device)

            optimizer.zero_grad()
            pa, pp, pr = model(xc, xn, xs, l)

            B    = xc.shape[0]
            last = (l - 1).long()

            # target 對齊最後一步
            ya_last = ya[torch.arange(B), last]
            yp_last = yp[torch.arange(B), last]

            loss_act = loss_fn_act(pa, ya_last)
            loss_pt  = loss_fn_pt (pp, yp_last)
            loss_rly = loss_fn_bin(pr, yr)

            # 任何一個 loss 是 NaN 就跳過（正常情況不應發生）
            if loss_act.isnan() or loss_pt.isnan() or loss_rly.isnan():
                continue

            # 對齊評分公式的損失權重
            loss = 0.4 * loss_act + 0.4 * loss_pt + 0.2 * loss_rly
            loss.backward()

            # 梯度裁剪閾值 5.0（原版 1.0 太激進，導致梯度爆炸警告誤觸發）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses["total"] += loss.item()
            losses["act"]   += loss_act.item()
            losses["pt"]    += loss_pt.item()
            losses["rly"]   += loss_rly.item()
            nb += 1

        scheduler.step()

        if nb == 0:
            print(f"Ep {ep:3d} | ⚠️ 所有批次都跳過（NaN）")
            continue

        avg = {k: v / nb for k, v in losses.items()}
        val_score, val_f1a, val_f1p, val_auc = evaluate(
            model, val_loader, device, act_classes, pt_classes
        )
        ep_time = time.time() - ep_start
        eta     = format_time(((time.time() - start) / ep) * (args.epochs - ep))

        if val_score > best_score:
            best_score = val_score
            no_improve = 0
            save_best_model(model, optimizer, ep, best_score, args.model_save, str(device))
            mark = " ✓ Best"
        else:
            no_improve += 1
            mark = ""

        print(
            f"Ep {ep:3d}/{args.epochs} | "
            f"Loss={avg['total']:.4f} [act={avg['act']:.3f} pt={avg['pt']:.3f} rly={avg['rly']:.3f}] | "
            f"Val={val_score:.4f}{mark} "
            f"[F1a={val_f1a:.3f} F1p={val_f1p:.3f} AUC={val_auc:.3f}] | "
            f"LR={optimizer.param_groups[0]['lr']:.1e} | "
            f"⏱{format_time(ep_time)} ETA:{eta}"
        )

        if no_improve >= patience:
            print(f"\n⏸ Early stopping at epoch {ep}")
            break

    print(f"\n✅ 訓練完成！最佳 Overall Score: {best_score:.4f}")

    # ── 推論 ──────────────────────────────────────────────────────────────────
    model, optimizer, _, _ = load_best_model(model, optimizer, args.model_save, device)
    model.eval()
    results = []

    with torch.no_grad():
        for rid, g in test_df.groupby("rally_uid"):
            phase = compute_strike_phase(g).reshape(-1, 1)
            xc = torch.tensor(encode_cat(g)[None],        dtype=torch.long,    device=device)
            xn = torch.tensor(encode_num(g)[None],        dtype=torch.float32, device=device)
            xs = torch.tensor(encode_sem(g, phase)[None], dtype=torch.float32, device=device)
            l  = torch.tensor([len(g)],                   dtype=torch.long,    device=device)

            pa, pp, pr = model(xc, xn, xs, l)

            results.append({
                "rally_uid":      rid,
                "serverGetPoint": torch.sigmoid(pr[0]).item(),  # 機率值（AUC-ROC 需要）
                "pointId":        pt_classes[pp[0].argmax().item()],
                "actionId":       act_classes[pa[0].argmax().item()],
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"✅ 結果已儲存至 {args.out}（{len(results)} 筆）")
    print(f"\nserverGetPoint 機率分布:")
    print(out_df["serverGetPoint"].describe().round(4))
    print(f"\nactionId 分布:")
    print(out_df["actionId"].value_counts())
    print(f"\npointId 分布:")
    print(out_df["pointId"].value_counts())


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",      default="train.csv")
    p.add_argument("--test",       default="test.csv")
    p.add_argument("--out",        default="submission_lstm_baseline.csv")
    p.add_argument("--model_save", default="best_model.pth")
    p.add_argument("--epochs",     type=int,   default=120)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    main(p.parse_args())
