"""
Baseline PyTorch MLP — capigradmie
===================================
Overfitting oldini olish:
  1. Dropout
  2. BatchNorm
  3. Early stopping (val loss asosida)
  4. Learning rate scheduler
  5. Walk-forward validation (time-aware)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# ──────────────────────────────────────────────
# 1. MODEL ARXITEKTURASI
# ──────────────────────────────────────────────

class TradingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list = [256, 128, 64], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))  # Binary: long vs short
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────
# 2. EARLY STOPPING
# ──────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = np.inf
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience  # True = stop

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ──────────────────────────────────────────────
# 3. TRAIN LOOP
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / total, correct / total, f1


# ──────────────────────────────────────────────
# 4. WALK-FORWARD VALIDATION
# ──────────────────────────────────────────────

def walk_forward(df, features, device, n_epochs=100, batch_size=512):
    windows = [
        ('2023-01-01', '2024-01-01', '2024-01-01', '2024-07-01'),
        ('2023-01-01', '2024-07-01', '2024-07-01', '2025-01-01'),
        ('2023-01-01', '2025-01-01', '2025-01-01', '2025-07-01'),
        ('2023-01-01', '2025-07-01', '2025-07-01', '2026-04-07'),
    ]

    m15 = pd.read_csv('data/btcusdt_m15.csv', index_col='open_time', parse_dates=True)
    m15 = m15[~m15.index.duplicated(keep='last')].sort_index()

    all_results = []

    for fold, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}: Train {tr_s[:7]}→{tr_e[:7]}  |  Test {te_s[:7]}→{te_e[:7]}")

        train_df = df[(df.index >= tr_s) & (df.index < tr_e)]
        test_df  = df[(df.index >= te_s) & (df.index < te_e)]

        # Val: train ning oxirgi 20%
        val_size  = int(len(train_df) * 0.2)
        val_df    = train_df.iloc[-val_size:]
        train_df2 = train_df.iloc[:-val_size]

        print(f"  Train: {len(train_df2):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df2[features].values).astype(np.float32)
        X_val   = scaler.transform(val_df[features].values).astype(np.float32)
        X_test  = scaler.transform(test_df[features].values).astype(np.float32)

        y_train = train_df2['label_bin'].values.astype(np.float32)
        y_val   = val_df['label_bin'].values.astype(np.float32)
        y_test  = test_df['label_bin'].values.astype(np.float32)

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=batch_size
        )

        # Model
        model = TradingMLP(input_dim=len(features), hidden=[256, 128, 64], dropout=0.3).to(device)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion  = nn.BCEWithLogitsLoss()
        stopper    = EarlyStopping(patience=12)

        # Training
        print(f"  {'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10} {'F1':>8}")
        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_loss, vl_acc, vl_f1 = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(vl_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>5} {tr_loss:>10.4f} {tr_acc:>10.3f} {vl_loss:>10.4f} {vl_acc:>10.3f} {vl_f1:>8.3f}")

            if stopper.step(vl_loss, model):
                print(f"  Early stop at epoch {epoch}, best val_loss={stopper.best_loss:.4f}")
                break

        stopper.restore(model)

        # Test evaluation
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test).to(device)
            logits = model(X_t)
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, zero_division=0)
        print(f"\n  TEST → Accuracy: {acc*100:.1f}%  |  F1: {f1:.3f}")

        # PnL backtest
        test_copy = test_df.copy()
        test_copy['pred'] = np.where(preds == 1, 1, -1)
        capital = 1000.0
        position_end = None
        HALF_KELLY = 0.081
        trades = []

        for ts, row in test_copy.iterrows():
            if position_end is not None and ts < position_end:
                continue
            try:
                entry = m15.loc[ts, 'close']
                fi    = m15.index.get_loc(ts) + 8
                if fi >= len(m15): continue
                exit_ = m15.iloc[fi]['close']
            except: continue
            ret = row['pred'] * (exit_/entry - 1) - 0.0006
            capital += capital * HALF_KELLY * ret
            trades.append(ret)
            position_end = m15.index[min(fi, len(m15)-1)]

        t = np.array(trades)
        total_ret = (capital/1000 - 1) * 100
        print(f"  PnL → WR: {(t>0).mean()*100:.1f}%  Return: {total_ret:.1f}%  Trades: {len(t)}")
        all_results.extend(trades)

        # Modelni saqlash
        Path('model').mkdir(exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_std':  scaler.scale_,
            'features':    features,
            'input_dim':   len(features),
        }, f'model/mlp_fold{fold+1}.pt')

    # Umumiy natija
    all_t = np.array(all_results)
    print(f"\n{'='*50}")
    print(f"JAMI WALK-FORWARD NATIJA:")
    print(f"  Avg PnL/trade : {all_t.mean()*100:.3f}%")
    print(f"  Win rate      : {(all_t>0).mean()*100:.1f}%")


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = pd.read_parquet('data/dataset_v2.parquet')
    df = ds[ds['label'] != 0].copy()
    df['label_bin'] = (df['label'] == 1).astype(int)

    drop = ['last_swing_high', 'last_swing_low',
            'h1_last_swing_high', 'h1_last_swing_low']
    features = [c for c in df.columns
                if c not in ['label', 'label_bin'] and c not in drop]

    print(f"Features: {len(features)} | Samples: {len(df):,}")
    walk_forward(df, features, device, n_epochs=100, batch_size=512)
