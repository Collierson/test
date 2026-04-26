"""
乒乓球戰術預測 v3 — 結合代號語義的特徵工程強化版
=======================================================
根據「各代號的意思.xlsx」加入以下領域知識：

【新增特徵】
1. action_type_id    : 將 actionId 對應到球種類型（Attack/Control/Defensive/Serve）
2. point_row / point_col: 將九宮格 pointId 拆解為行列座標（空間語義）
3. spin_x / spin_y   : 旋轉方向向量化（上旋→+1, 下旋→-1, 側旋→左右 ±0.5）
4. is_serve / is_receive: strikeId 1/2 的 one-hot（發球vs接發球）
5. tactical_ctx      : handId × positionId 合成戰術情境（正手近台/反手長球等）
6. momentum          : 滾動得分差（最近3拍的 scoreDiff 趨勢）
7. strike_phase      : 序列中的相對位置（前/中/後段）

【模型改進】
- 三個任務頭各自接 task-specific cross-attention（用序列資訊增強最後一步）
- ActionId 頭加入 action_type 輔助分類器（multi-task 細化）
- PointId 頭加入位置座標回歸分支（輔助訓練，提升空間感知）
- serverGetPoint 頭使用全序列池化（mean+max），而非只用最後一步

【修正保留】
✅ target 對齊 last_valid_idx
✅ serverGetPoint 輸出機率值
✅ 損失權重 0.4/0.4/0.2 對齊評分公式
✅ 驗證集 Early Stopping
"""

import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score
import time
from datetime import timedelta
import os

# ── 隨機種子 ─────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 代號語義映射（來自「各代號的意思.xlsx」）
# ══════════════════════════════════════════════════════════════════════════════

# actionId → action_type: 0=Serve, 1=Attack, 2=Control, 3=Defensive
ACTION_TYPE_MAP = {
    0: -1,   # 無/其他
    1: 1,    # 拉球       Attack
    2: 1,    # 反拉       Attack
    3: 1,    # 殺球       Attack
    4: 1,    # 擰球       Attack
    5: 1,    # 快帶       Attack
    6: 1,    # 推擠       Attack
    7: 1,    # 挑撥       Attack
    8: 2,    # 拱球       Control
    9: 2,    # 磕球       Control
    10: 2,   # 搓球       Control
    11: 2,   # 擺短       Control
    12: 3,   # 削球       Defensive
    13: 3,   # 擋球       Defensive
    14: 3,   # 放高球     Defensive
    15: 0,   # 傳統       Serve
    16: 0,   # 勾手       Serve
    17: 0,   # 逆旋轉     Serve
    18: 0,   # 下蹲式     Serve
}

# pointId → (row, col)：九宮格座標 (1-based 轉 0-based)
# 九宮格佈局（從對方視角）：
#   7 8 9
#   4 5 6
#   1 2 3
POINT_COORD_MAP = {
    0: (0, 0),    # 無落點（網/出界）→ 中心
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}

# spinId → (vertical, horizontal)：旋轉向量
SPIN_VEC_MAP = {
    0: (0.0,  0.0),   # 無
    1: (1.0,  0.0),   # 上旋
    2: (-1.0, 0.0),   # 下旋
    3: (0.0,  0.0),   # 不旋
    4: (0.5,  1.0),   # 側上旋
    5: (-0.5, 1.0),   # 側下旋
}

# 類別特徵（直接 embedding）
CAT_FEATURES = ["sex", "handId", "strengthId", "spinId",
                "pointId", "actionId", "positionId", "strikeId"]

# 數值特徵（標準化）
NUM_FEATURES = ["scoreSelf", "scoreOther", "strikeNumber", "scoreDiff"]

# 新增語義特徵（數值型）
SEM_FEATURES = [
    "point_row", "point_col",       # 九宮格座標
    "spin_v", "spin_h",             # 旋轉向量
    "is_serve", "is_receive",       # 發球/接發球
    "tactical_ctx",                 # 正手/反手 × 站位合成
    "action_is_attack", "action_is_control", "action_is_defensive", "action_is_serve",  # 球種類型 one-hot
    "strike_phase",                 # 序列位置（0=開局 ~ 1=末段）
]

PAD_TOKEN = 0
N_ACT_TYPES = 4  # Serve/Attack/Control/Defensive


# ══════════════════════════════════════════════════════════════════════════════
# 特徵工程
# ══════════════════════════════════════════════════════════════════════════════
def add_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 九宮格座標
    df["point_row"] = df["pointId"].map(lambda x: POINT_COORD_MAP.get(x, (0, 0))[0])
    df["point_col"] = df["pointId"].map(lambda x: POINT_COORD_MAP.get(x, (0, 0))[1])

    # 旋轉向量
    df["spin_v"] = df["spinId"].map(lambda x: SPIN_VEC_MAP.get(x, (0.0, 0.0))[0])
    df["spin_h"] = df["spinId"].map(lambda x: SPIN_VEC_MAP.get(x, (0.0, 0.0))[1])

    # 發球/接發球 flag（strikeId: 1=發球, 2=接發球）
    df["is_serve"]   = (df["strikeId"] == 1).astype(float)
    df["is_receive"] = (df["strikeId"] == 2).astype(float)

    # 戰術情境：handId(1=正手, 2=反手) × positionId(1=左, 2=中, 3=右)
    # 合成為 0~5 的情境碼（標準化到 0-1）
    df["tactical_ctx"] = ((df["handId"].clip(0, 2) - 1) * 3 +
                          (df["positionId"].clip(0, 3) - 1)).clip(0, 5) / 5.0

    # 球種類型 one-hot
    act_type = df["actionId"].map(lambda x: ACTION_TYPE_MAP.get(x, -1))
    df["action_is_serve"]     = (act_type == 0).astype(float)
    df["action_is_attack"]    = (act_type == 1).astype(float)
    df["action_is_control"]   = (act_type == 2).astype(float)
    df["action_is_defensive"] = (act_type == 3).astype(float)

    # 序列相對位置（在 groupby 後才能計算，這裡先填 0，後面 per-rally 再更新）
    df["strike_phase"] = 0.0

    return df


def compute_strike_phase(seq_df: pd.DataFrame) -> np.ndarray:
    """計算每一拍在 rally 中的相對位置 (0.0 ~ 1.0)"""
    n = len(seq_df)
    if n <= 1:
        return np.zeros(n)
    return np.arange(n) / (n - 1)


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

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, i):
        return (self.X_cat[i], self.X_num[i], self.X_sem[i],
                self.yA[i], self.yP[i], self.yR[i], self.L[i])


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', ignore_index=ignore_index)

    def forward(self, pred, target):
        ce    = self.ce(pred, target)
        p     = torch.exp(-ce)
        focal = self.alpha * (1 - p) ** self.gamma * ce
        valid = (target != self.ignore_index)
        return focal[valid].mean() if valid.any() else torch.tensor(0.0, device=pred.device)


# ══════════════════════════════════════════════════════════════════════════════
# Task-Specific Cross-Attention
# ══════════════════════════════════════════════════════════════════════════════
class TaskCrossAttention(nn.Module):
    """用一個可學習的 task query 對整個序列做 cross-attention，萃取任務相關資訊"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.task_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq, key_padding_mask=None):
        # seq: [B, T, D]
        B = seq.shape[0]
        q = self.task_query.expand(B, -1, -1)          # [B, 1, D]
        out, _ = self.attn(q, seq, seq, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))                # [B, D]


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
class SemanticMultiTaskLSTM(nn.Module):
    def __init__(self, cat_dims, num_dim, sem_dim, n_act, n_pt,
                 emb_dim=32, hidden=256, num_layers=3, dropout=0.35):
        super().__init__()

        # 類別 Embedding
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in cat_dims
        ])

        lstm_in = len(cat_dims) * emb_dim + num_dim + sem_dim
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_in, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU()
        )

        self.lstm = nn.LSTM(
            hidden * 2, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(hidden * 2)

        # Task-specific Cross-Attention（每個任務獨立萃取序列資訊）
        self.act_cross_attn = TaskCrossAttention(hidden * 2, num_heads=4)
        self.pt_cross_attn  = TaskCrossAttention(hidden * 2, num_heads=4)
        self.rly_cross_attn = TaskCrossAttention(hidden * 2, num_heads=4)

        fusion_dim = hidden * 2 + hidden * 2  # last_step + cross_attn

        # ── ActionId 頭：主分類 + action_type 輔助分類器 ──────────────────
        self.act_shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),     nn.LayerNorm(hidden),     nn.GELU(), nn.Dropout(dropout),
        )
        self.act_out      = nn.Linear(hidden, n_act)
        self.act_type_out = nn.Linear(hidden, N_ACT_TYPES)   # 輔助：Serve/Attack/Control/Defensive

        # ── PointId 頭：主分類 + 座標回歸輔助 ────────────────────────────
        self.pt_shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),     nn.LayerNorm(hidden),     nn.GELU(), nn.Dropout(dropout),
        )
        self.pt_out      = nn.Linear(hidden, n_pt)
        self.pt_coord_out = nn.Linear(hidden, 2)   # 輔助：預測 (row, col) 座標

        # ── ServerGetPoint 頭：mean+max 池化 + last_step ─────────────────
        self.rly_pool_proj = nn.Linear(hidden * 2 * 2, hidden * 2)  # mean+max concat
        self.rly_head = nn.Sequential(
            nn.Linear(hidden * 2 + hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),              nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, X_cat, X_num, X_sem, lengths):
        B, T = X_cat.shape[:2]

        # Embedding + concat
        embs = [emb(X_cat[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(embs + [X_num, X_sem], dim=-1)
        x = self.input_proj(x)

        # LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=T)
        lstm_out = self.lstm_norm(lstm_out)

        # Padding mask（True 表示 padding 位置，MultiheadAttention 的 key_padding_mask 格式）
        pad_mask = (X_cat[:, :, 0] == PAD_TOKEN)   # [B, T]，True=padding

        # 最後一個有效步驟
        last_idx  = (lengths - 1).long()
        last_step = lstm_out[torch.arange(B), last_idx]   # [B, D]

        # ── ActionId ─────────────────────────────────────────────────────
        act_ctx    = self.act_cross_attn(lstm_out, key_padding_mask=pad_mask)
        act_feat   = self.act_shared(torch.cat([last_step, act_ctx], dim=-1))
        act_out    = self.act_out(act_feat)
        act_type   = self.act_type_out(act_feat)

        # ── PointId ──────────────────────────────────────────────────────
        pt_ctx     = self.pt_cross_attn(lstm_out, key_padding_mask=pad_mask)
        pt_feat    = self.pt_shared(torch.cat([last_step, pt_ctx], dim=-1))
        pt_out     = self.pt_out(pt_feat)
        pt_coord   = self.pt_coord_out(pt_feat)            # (row, col) 回歸

        # ── ServerGetPoint：使用全序列池化 ───────────────────────────────
        rly_ctx    = self.rly_cross_attn(lstm_out, key_padding_mask=pad_mask)
        # mean + max pooling（遮掉 padding）
        mask_f     = (~pad_mask).float().unsqueeze(-1)     # [B, T, 1]
        seq_mean   = (lstm_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        seq_max    = lstm_out.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(1).values
        pool_feat  = self.rly_pool_proj(torch.cat([seq_mean, seq_max], dim=-1))
        rly_out    = self.rly_head(torch.cat([pool_feat, rly_ctx], dim=-1)).squeeze(-1)

        return act_out, act_type, pt_out, pt_coord, rly_out


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


def get_focal_loss_weights(labels, n_classes):
    valid = labels[labels != -1]
    if len(valid) == 0:
        return torch.ones(n_classes)
    present = np.unique(valid)
    raw_w   = compute_class_weight('balanced', classes=present, y=valid)
    full_w  = np.ones(n_classes)
    for idx, l in enumerate(present):
        full_w[l] = raw_w[idx]
    return torch.tensor(full_w, dtype=torch.float32)


def compute_sample_weights(labels_2d, n_classes):
    valid_mask = (labels_2d != -1)
    all_valid  = labels_2d[valid_mask]
    if len(all_valid) == 0:
        return np.ones(len(labels_2d))
    present = np.unique(all_valid)
    raw_w   = compute_class_weight('balanced', classes=present, y=all_valid)
    full_w  = np.ones(n_classes)
    for idx, l in enumerate(present):
        full_w[l] = raw_w[idx]
    weights = []
    for row in labels_2d:
        valid_row = row[row != -1]
        w = np.mean([full_w[l] for l in valid_row]) if len(valid_row) > 0 else 1.0
        weights.append(w)
    return np.array(weights)


# ══════════════════════════════════════════════════════════════════════════════
# 評估（競賽評分公式）
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, device):
    model.eval()
    all_ya, all_pa = [], []
    all_yp, all_pp = [], []
    all_yr, all_pr = [], []

    with torch.no_grad():
        for xc, xn, xs, ya, yp, yr, l in loader:
            xc = xc.to(device); xn = xn.to(device); xs = xs.to(device)
            ya = ya.to(device); yp = yp.to(device)
            yr = yr.to(device).float(); l = l.to(device)

            pa, _, pp, _, pr = model(xc, xn, xs, l)
            B    = xc.shape[0]
            last = (l - 1).long()

            ya_last = ya[torch.arange(B), last]
            yp_last = yp[torch.arange(B), last]

            act_valid = ya_last != -1
            pt_valid  = yp_last != -1

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

    overall = 0.4 * f1_act + 0.4 * f1_pt + 0.2 * auc
    return overall, f1_act, f1_pt, auc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    print("=" * 80)
    print("乒乓球戰術預測 v3 — 代號語義特徵強化版")
    print("=" * 80)

    train_df = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test_df  = pd.read_csv(args.test) .sort_values(["rally_uid", "strikeNumber"])

    # ── 基礎特徵 ────────────────────────────────────────────────────────────
    train_df["scoreDiff"] = train_df["scoreSelf"] - train_df["scoreOther"]
    test_df ["scoreDiff"] = test_df ["scoreSelf"] - test_df ["scoreOther"]

    # ── 語義特徵 ────────────────────────────────────────────────────────────
    train_df = add_semantic_features(train_df)
    test_df  = add_semantic_features(test_df)

    # ── 數值標準化（含語義特徵中的連續值）──────────────────────────────────
    all_num_cols = NUM_FEATURES + ["point_row", "point_col", "spin_v", "spin_h", "tactical_ctx"]
    for col in all_num_cols:
        m, s = train_df[col].mean(), train_df[col].std()
        train_df[col] = (train_df[col] - m) / (s + 1e-7)
        test_df [col] = (test_df [col] - m) / (s + 1e-7)

    # ── 類別特徵編碼 ────────────────────────────────────────────────────────
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
        data = np.stack([df[c].values for c in base_cols], axis=1)
        ph = phase_arr if phase_arr is not None else np.zeros((len(df), 1))
        if ph.ndim == 1: ph = ph.reshape(-1, 1)
        return np.concatenate([data, ph], axis=1)

    # ── 序列構建 ─────────────────────────────────────────────────────────────
    X_cat_list, X_num_list, X_sem_list = [], [], []
    yA_list, yP_list, yR_list, L_list  = [], [], [], []

    for rid, g in train_df.groupby("rally_uid"):
        if len(g) < 2:
            continue
        phase = compute_strike_phase(g).reshape(-1, 1)
        xc    = encode_cat(g)
        xn    = encode_num(g)
        xs    = encode_sem(g, phase)

        X_cat_list.append(xc[:-1])
        X_num_list.append(xn[:-1])
        X_sem_list.append(xs[:-1])
        yA_list.append(g["actionId"].values[1:])
        yP_list.append(g["pointId"].values[1:])
        yR_list.append(int(g["serverGetPoint"].iloc[0]))
        L_list.append(len(g) - 1)

    MAXLEN = max(L_list)
    SEM_DIM = X_sem_list[0].shape[1]

    def pad2d(lst, pad_val=0):
        return np.array([np.pad(a, ((0, MAXLEN - len(a)), (0, 0)), constant_values=pad_val) for a in lst])

    def pad1d(lst, pad_val=-1):
        return np.array([np.pad(a, (0, MAXLEN - len(a)), constant_values=pad_val) for a in lst])

    X_cat_all = pad2d(X_cat_list)
    X_num_all = pad2d(X_num_list)
    X_sem_all = pad2d(X_sem_list, pad_val=0.0)
    yA_all    = pad1d(yA_list)
    yP_all    = pad1d(yP_list)
    yR_all    = np.array(yR_list)
    L_all     = np.array(L_list)

    # ── 類別映射 ─────────────────────────────────────────────────────────────
    act_classes = np.sort(train_df["actionId"].unique())
    pt_classes  = np.array([p for p in np.sort(train_df["pointId"].unique()) if p != 0])
    act_map     = {v: i for i, v in enumerate(act_classes)}
    pt_map      = {v: i for i, v in enumerate(pt_classes)}

    # action_type 映射（act_map 索引 → type 索引）
    act_type_target = np.full(len(act_classes), -1, dtype=int)
    for act_val, act_idx in act_map.items():
        t = ACTION_TYPE_MAP.get(act_val, -1)
        act_type_target[act_idx] = t if t >= 0 else -1

    yA_flat = np.vectorize(lambda x: act_map.get(x, -1))(yA_all)
    yP_flat = np.vectorize(lambda x: pt_map.get(x, -1))(yP_all)

    # pointId → 座標 target（用於輔助回歸）
    pt_coord_map = {v: i for i, v in enumerate(pt_classes)}  # same as pt_map
    def get_pt_coord(pt_idx):
        """pt_idx (已映射) → (row, col) 浮點座標"""
        if pt_idx < 0 or pt_idx >= len(pt_classes):
            return np.array([-1.0, -1.0])
        raw_pid = pt_classes[pt_idx]
        row, col = POINT_COORD_MAP.get(raw_pid, (0, 0))
        return np.array([row / 2.0, col / 2.0])  # 標準化到 0-1

    # 診斷
    last_ya = yA_flat[np.arange(len(L_all)), L_all - 1]
    last_yp = yP_flat[np.arange(len(L_all)), L_all - 1]
    print(f"\n📊 資料統計：")
    print(f"  • 訓練樣本數: {len(X_cat_all)}")
    print(f"  • 最大序列長度: {MAXLEN}")
    print(f"  • Action 類別數: {len(act_map)} | Point 類別數: {len(pt_map)}")
    print(f"  • 語義特徵維度: {SEM_DIM}")
    print(f"  • 最後一拍 ActionId 有效標籤: {(last_ya!=-1).sum()}/{len(L_all)}")
    print(f"  • 最後一拍 PointId  有效標籤: {(last_yp!=-1).sum()}/{len(L_all)}")

    # ── Loss 權重 ────────────────────────────────────────────────────────────
    act_weights = get_focal_loss_weights(yA_flat, len(act_map))
    pt_weights  = get_focal_loss_weights(yP_flat, len(pt_map))

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

    train_loader = DataLoader(make_ds(tr_idx), batch_size=args.batch, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(make_ds(va_idx), batch_size=args.batch, shuffle=False, num_workers=0)

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

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  • 參數數: {total_p:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=5e-6
    )

    # Loss 函式
    loss_fn_act      = FocalLoss(alpha=0.75, gamma=2.0, weight=act_weights.to(device), ignore_index=-1)
    loss_fn_pt       = FocalLoss(alpha=0.75, gamma=2.0, weight=pt_weights.to(device),  ignore_index=-1)
    loss_fn_act_type = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn_coord    = nn.MSELoss()
    loss_fn_bin      = nn.BCEWithLogitsLoss()

    # act_type_target 放到 device
    act_type_tgt = torch.tensor(act_type_target, dtype=torch.long, device=device)

    print("\n" + "=" * 80)
    print("開始訓練 v3")
    print("=" * 80)
    print(f"配置: Batch={args.batch}, LR={args.lr}, Epochs={args.epochs}")
    print("新增:")
    print("  🆕 語義特徵: 九宮格座標/旋轉向量/發球接發/戰術情境/球種類型/序列位置")
    print("  🆕 Task-specific Cross-Attention（每個任務獨立萃取序列資訊）")
    print("  🆕 ActionId 輔助分類器（action_type: Serve/Attack/Control/Defensive）")
    print("  🆕 PointId 座標回歸輔助（空間語義監督）")
    print("  🆕 ServerGetPoint 使用 mean+max 全序列池化")
    print("損失: 0.4*act + 0.4*pt + 0.2*rly + 0.05*act_type + 0.05*pt_coord\n")

    best_score = -1.0
    patience = 20
    no_improve = 0
    start = time.time()

    for ep in range(1, args.epochs + 1):
        ep_start = time.time()
        model.train()
        losses = {"total": 0, "act": 0, "pt": 0, "rly": 0, "aux": 0}
        nb = 0

        for xc, xn, xs, ya, yp, yr, l in train_loader:
            xc = xc.to(device); xn = xn.to(device); xs = xs.to(device)
            ya = ya.to(device); yp = yp.to(device)
            yr = yr.to(device).float(); l = l.to(device)

            optimizer.zero_grad()
            pa, pa_type, pp, pp_coord, pr = model(xc, xn, xs, l)

            B    = xc.shape[0]
            last = (l - 1).long()
            ya_last = ya[torch.arange(B), last]
            yp_last = yp[torch.arange(B), last]

            # ── 主任務損失 ────────────────────────────────────────────────
            loss_act = loss_fn_act(pa, ya_last)
            loss_pt  = loss_fn_pt(pp,  yp_last)
            loss_rly = loss_fn_bin(pr, yr)

            # ── 輔助損失 1：action_type 分類 ─────────────────────────────
            act_valid_mask = (ya_last != -1)
            if act_valid_mask.any():
                ya_last_valid = ya_last[act_valid_mask]
                pa_type_valid = pa_type[act_valid_mask]
                at_target     = act_type_tgt[ya_last_valid.clamp(0, len(act_type_tgt)-1)]
                loss_act_type = loss_fn_act_type(pa_type_valid, at_target)
            else:
                loss_act_type = torch.tensor(0.0, device=device)

            # ── 輔助損失 2：pointId 座標回歸 ─────────────────────────────
            pt_valid_mask = (yp_last != -1)
            if pt_valid_mask.any():
                yp_last_valid  = yp_last[pt_valid_mask]
                pp_coord_valid = pp_coord[pt_valid_mask]
                coord_targets  = torch.tensor(
                    np.array([get_pt_coord(i) for i in yp_last_valid.cpu().numpy()]),
                    dtype=torch.float32, device=device
                )
                loss_coord = loss_fn_coord(pp_coord_valid, coord_targets)
            else:
                loss_coord = torch.tensor(0.0, device=device)

            # ── 綜合損失（主任務對齊評分公式，輔助任務小權重）────────────
            loss = (0.40 * loss_act +
                    0.40 * loss_pt  +
                    0.20 * loss_rly +
                    0.05 * loss_act_type +
                    0.05 * loss_coord)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses["total"] += loss.item()
            losses["act"]   += loss_act.item()
            losses["pt"]    += loss_pt.item()
            losses["rly"]   += loss_rly.item()
            losses["aux"]   += (loss_act_type + loss_coord).item()
            nb += 1

        scheduler.step()

        avg = {k: v / nb for k, v in losses.items()}
        val_score, val_f1a, val_f1p, val_auc = evaluate(model, val_loader, device)
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
            f"Loss={avg['total']:.4f} [act={avg['act']:.3f} pt={avg['pt']:.3f} "
            f"rly={avg['rly']:.3f} aux={avg['aux']:.3f}] | "
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
            xc = torch.tensor(encode_cat(g)[None],          dtype=torch.long,   device=device)
            xn = torch.tensor(encode_num(g)[None],          dtype=torch.float32, device=device)
            xs = torch.tensor(encode_sem(g, phase)[None],   dtype=torch.float32, device=device)
            l  = torch.tensor([len(g)],                     dtype=torch.long,   device=device)

            pa, _, pp, _, pr = model(xc, xn, xs, l)

            results.append({
                "rally_uid":      rid,
                "serverGetPoint": torch.sigmoid(pr[0]).item(),   # 機率值
                "pointId":        pt_classes[pp[0].argmax().item()],
                "actionId":       act_classes[pa[0].argmax().item()],
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"✅ 預測已儲存至 {args.out}（{len(results)} 筆）")
    print("\nserverGetPoint 機率分布:")
    print(out_df["serverGetPoint"].describe().round(4))


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",      default="train.csv")
    p.add_argument("--test",       default="test.csv")
    p.add_argument("--out",        default="submission_lstm_baseline.csv")
    p.add_argument("--model_save", default="best_model.pth")
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    main(p.parse_args())
