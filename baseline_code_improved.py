import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import time
from datetime import timedelta
import sys
import os

# 設定隨機種子
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CAT_FEATURES = ["sex", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId", "strikeId"]
NUM_FEATURES = ["scoreSelf", "scoreOther", "strikeNumber", "scoreDiff"]
PAD_TOKEN = 0

class RallyDataset(Dataset):
    def __init__(self, X_cat, X_num, yA, yP, yR, L):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
    
    def __len__(self):
        return self.X_cat.shape[0]
    
    def __getitem__(self, i):
        return self.X_cat[i], self.X_num[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]

class FocalLoss(nn.Module):
    """
    Focal Loss 用於處理類別不平衡問題
    對簡單樣本降權，對難樣本加權
    公式：FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, weight=None, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', 
                                              weight=self.weight, ignore_index=self.ignore_index)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, values, keys, query, mask):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(keys)
        V = self.value(values)
        
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        context = context.transpose(1, 2).reshape(batch_size, -1, self.hidden_dim)
        output = self.fc_out(context)
        
        return output

class ImprovedMultiTaskLSTM(nn.Module):
    def __init__(self, cat_dims, num_dim, n_act, n_pt, emb_dim=32, hidden=256, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in cat_dims])
        
        lstm_input_dim = len(cat_dims) * emb_dim + num_dim
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = MultiHeadAttention(hidden * 2, num_heads=4)
        self.layer_norm = nn.LayerNorm(hidden * 2)
        
        # 加深 ActionId 預測頭（改善 ActionId 的主要原因）
        self.act_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_act)
        )
        
        self.pt_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_pt)
        )
        
        self.rly_head = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, X_cat, X_num, lengths):
        embeddings = [emb(X_cat[:,:,i]) for i, emb in enumerate(self.embs)]
        x_cat = torch.cat(embeddings, dim=-1)
        x = torch.cat([x_cat, X_num], dim=-1)
        
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=X_cat.size(1))
        
        lstm_out = self.layer_norm(lstm_out)
        
        mask = (X_cat[:,:,0] != PAD_TOKEN).float()
        attn_out = self.attention(lstm_out, lstm_out, lstm_out, mask)
        
        batch_size = lstm_out.shape[0]
        last_valid_idx = (lengths - 1).long()
        
        last_outputs = lstm_out[torch.arange(batch_size), last_valid_idx]
        last_attn = attn_out[torch.arange(batch_size), last_valid_idx]
        
        act_out = self.act_head(last_outputs)
        pt_out = self.pt_head(last_outputs)
        rly_out = self.rly_head(self.drop(last_attn)).squeeze(1)
        
        return act_out, pt_out, rly_out

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def save_best_model(model, optimizer, epoch, best_loss, filepath, device_name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'device': device_name
    }
    torch.save(checkpoint, filepath)
    print(f"💾 最佳模型已儲存至 {filepath}")

def load_best_model(model, optimizer, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    print(f"📂 載入最佳模型 (Epoch {epoch}, Loss: {best_loss:.4f}) 自 {filepath}")
    return model, optimizer, epoch, best_loss

def get_weighted_sampler(labels, num_samples=None):
    """
    創建加權隨機採樣器，對少數類別進行過採樣
    """
    from collections import Counter
    class_counts = Counter(labels[labels != -1])
    class_weights = {}
    min_count = min(class_counts.values())
    
    for cls, count in class_counts.items():
        class_weights[cls] = min_count / (count + 1e-8)
    
    sample_weights = np.array([class_weights.get(int(label), 1.0) for label in labels])
    
    if num_samples is None:
        num_samples = len(labels)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    return sampler

def main(args):
    print("=" * 80)
    print("正在準備資料與特徵工程...")
    print("=" * 80)
    
    train = pd.read_csv(args.train).sort_values(["rally_uid","strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid","strikeNumber"])

    train["scoreDiff"] = train["scoreSelf"] - train["scoreOther"]
    test["scoreDiff"] = test["scoreSelf"] - test["scoreOther"]

    for col in NUM_FEATURES:
        m, s = train[col].mean(), train[col].std()
        train[col] = (train[col] - m) / (s + 1e-7)
        test[col] = (test[col] - m) / (s + 1e-7)

    cats = {c: pd.Categorical(train[c]).categories for c in CAT_FEATURES}
    
    def encode_data(df):
        cat_data, num_data = [], []
        for col in CAT_FEATURES:
            cat_data.append(pd.Categorical(df[col], categories=cats[col]).codes + 1)
        for col in NUM_FEATURES:
            num_data.append(df[col].values)
        return np.stack(cat_data, axis=1), np.stack(num_data, axis=1)

    X_cat_list, X_num_list, yA_list, yP_list, yR_list, L_list = [], [], [], [], [], []
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2:
            continue
        xc, xn = encode_data(g)
        X_cat_list.append(xc[:-1])
        X_num_list.append(xn[:-1])
        yA_list.append(g["actionId"].values[1:])
        yP_list.append(g["pointId"].values[1:])
        yR_list.append(int(g["serverGetPoint"].iloc[0]))
        L_list.append(len(xc)-1)

    MAXLEN = max(L_list)
    X_cat_all = np.array([np.pad(a, ((0, MAXLEN-len(a)), (0,0))) for a in X_cat_list])
    X_num_all = np.array([np.pad(a, ((0, MAXLEN-len(a)), (0,0))) for a in X_num_list])
    yA_all = np.array([np.pad(a, (0, MAXLEN-len(a)), constant_values=-1) for a in yA_list])
    yP_all = np.array([np.pad(a, (0, MAXLEN-len(a)), constant_values=-1) for a in yP_list])
    yR_all, L_all = np.array(yR_list), np.array(L_list)

    act_classes = np.sort(train["actionId"].unique())
    pt_classes = np.array([p for p in np.sort(train["pointId"].unique()) if p != 0])
    act_map = {v: i for i, v in enumerate(act_classes)}
    pt_map = {v: i for i, v in enumerate(pt_classes)}
    yA_flat = np.vectorize(lambda x: act_map.get(x, -1))(yA_all)
    yP_flat = np.vectorize(lambda x: pt_map.get(x, -1))(yP_all)

    print(f"\n📊 資料統計：")
    print(f"  • 訓練樣本數: {len(X_cat_all)}")
    print(f"  • 最大序列長度: {MAXLEN}")
    print(f"  • Action 類別數: {len(act_map)}")
    print(f"  • Point 類別數: {len(pt_map)}")

    def get_weights(labels, n_classes):
        valid_labels = labels[labels != -1]
        present_labels = np.unique(valid_labels)
        
        weights_raw = compute_class_weight(
            'balanced', 
            classes=present_labels, 
            y=valid_labels
        )
        
        full_weights = np.ones(n_classes)
        for idx, label in enumerate(present_labels):
            full_weights[label] = weights_raw[idx]
            
        return torch.tensor(full_weights, dtype=torch.float32)

    act_weights = get_weights(yA_flat, len(act_map))
    pt_weights  = get_weights(yP_flat, len(pt_map))

    tr_idx, va_idx = train_test_split(
        np.arange(len(L_all)), test_size=0.1, 
        stratify=yR_all, random_state=SEED
    )
    
    # 使用加權採樣器來處理 ActionId 類別不平衡
    action_sampler = get_weighted_sampler(yA_flat[tr_idx], len(tr_idx))
    
    train_dataset = RallyDataset(X_cat_all[tr_idx], X_num_all[tr_idx], yA_flat[tr_idx], 
                                 yP_flat[tr_idx], yR_all[tr_idx], L_all[tr_idx])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=action_sampler  # 使用加權採樣器
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  使用設備: {device}")
    
    model = ImprovedMultiTaskLSTM(
        cat_dims=[len(cats[c]) for c in CAT_FEATURES], 
        num_dim=len(NUM_FEATURES), 
        n_act=len(act_map), 
        n_pt=len(pt_map)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • 總參數數: {total_params:,}")
    print(f"  • 可訓練參數數: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    
    # 使用 Focal Loss 處理 ActionId 類別不平衡
    loss_fn_act = FocalLoss(gamma=2.0, weight=act_weights.to(device), ignore_index=-1)
    loss_fn_pt  = nn.CrossEntropyLoss(weight=pt_weights.to(device), ignore_index=-1)
    loss_fn_bin = nn.BCEWithLogitsLoss()

    print("\n" + "=" * 80)
    print(f"開始訓練 (進階訓練機制)")
    print("=" * 80)
    print(f"配置: Batch={args.batch}, LR={args.lr}, Epochs={args.epochs}")
    print(f"損失函數: Action=Focal Loss(γ=2.0), Point=CrossEntropy, Rally=BCEWithLogits")
    print(f"損失權重: Action=0.5, Point=0.3, Rally=0.2 (強化 ActionId)")
    print(f"採樣策略: WeightedRandomSampler (對少數類別過採樣)")
    print(f"最佳模型儲存位置: {args.model_save}\n")
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    start_time = time.time()
    
    for ep in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        act_loss_total = 0
        pt_loss_total = 0
        rly_loss_total = 0
        num_batches = 0
        
        for xc, xn, ya, yp, yr, l in train_loader:
            xc = xc.to(device)
            xn = xn.to(device)
            ya = ya.to(device)
            yp = yp.to(device)
            yr = yr.to(device).float()
            l = l.to(device)
            
            optimizer.zero_grad()
            
            pa, pp, pr = model(xc, xn, l)
            
            loss_act = loss_fn_act(pa, ya[:, 0])
            loss_pt = loss_fn_pt(pp, yp[:, 0])
            loss_rly = loss_fn_bin(pr, yr)
            
            # 調整權重：加強 ActionId
            loss = 0.5 * loss_act + 0.3 * loss_pt + 0.2 * loss_rly
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            act_loss_total += loss_act.item()
            pt_loss_total += loss_pt.item()
            rly_loss_total += loss_rly.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_act_loss = act_loss_total / num_batches
        avg_pt_loss = pt_loss_total / num_batches
        avg_rly_loss = rly_loss_total / num_batches
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        if ep > 0:
            avg_epoch_time = elapsed_time / ep
            remaining_epochs = args.epochs - ep
            estimated_remaining = avg_epoch_time * remaining_epochs
            eta_time = format_time(estimated_remaining)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_marker = " ✓ (Best)"
            save_best_model(model, optimizer, ep, best_loss, args.model_save, str(device))
        else:
            patience_counter += 1
            best_marker = ""
        
        print(f"Epoch {ep:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f}{best_marker} | "
              f"Act: {avg_act_loss:.4f} | "
              f"Pt: {avg_pt_loss:.4f} | "
              f"Rly: {avg_rly_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"⏱️  {format_time(epoch_time)} | "
              f"ETA: {eta_time}")
        
        if patience_counter >= patience:
            print(f"\n⏸️  Early stopping at epoch {ep} (patience={patience})")
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ 訓練完成！")
    print(f"   總耗時: {format_time(total_time)}")
    print(f"   最佳損失: {best_loss:.4f}")
    print(f"   最佳模型檔案: {args.model_save}")
    print("=" * 80)

    print("\n📂 載入最佳模型進行預測...")
    model, optimizer, best_ep, best_loss_val = load_best_model(model, optimizer, args.model_save, device)
    
    print("\n📝 生成預測...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            xc, xn = encode_data(g)
            xc_t = torch.tensor(xc[None, :], dtype=torch.long, device=device)
            xn_t = torch.tensor(xn[None, :], dtype=torch.float32, device=device)
            l_t = torch.tensor([len(g)], device=device, dtype=torch.long)
            
            pa, pp, pr = model(xc_t, xn_t, l_t)
            
            results.append({
                "rally_uid": rid,
                "serverGetPoint": torch.sigmoid(pr[0]).item(),
                "pointId": pt_classes[pp[0].argmax().item()],
                "actionId": act_classes[pa[0].argmax().item()]
            })

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"✅ 結果已儲存至 {args.out}")
    print(f"📊 預測樣本數: {len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out", default="submission_lstm_improved.csv")
    parser.add_argument("--model_save", default="best_model.pth", help="路徑用來儲存最佳模型")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    main(parser.parse_args())
