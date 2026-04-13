"""
PyTorch MLP v2 — confidence filter + to'g'ri saqlash
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

class TradingMLP(nn.Module):
    def __init__(self, input_dim, hidden=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_b).sum().item()
        total += len(y_b)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            total_loss += loss.item() * len(y_b)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_b).sum().item()
            total += len(y_b)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / total, correct / total, f1


def backtest(test_df, proba, features, m15, conf_threshold=0.70, half_kelly=0.081):
    """Confidence filter bilan backtest"""
    test_copy = test_df.copy()
    test_copy['prob']       = proba
    test_copy['confidence'] = np.maximum(proba, 1 - proba)
    test_copy['pred']       = np.where(proba > 0.5, 1, -1)

    # Confidence filter
    filtered = test_copy[test_copy['confidence'] >= conf_threshold]

    capital = 1000.0
    position_end = None
    max_capital = 1000.0
    max_dd = 0.0
    trades = []

    for ts, row in filtered.iterrows():
        if position_end is not None and ts < position_end:
            continue
        try:
            entry = m15.loc[ts, 'close']
            fi = m15.index.get_loc(ts) + 8
            if fi >= len(m15): continue
            exit_ = m15.iloc[fi]['close']
        except:
            continue

        ret = row['pred'] * (exit_ / entry - 1) - 0.0006
        capital += capital * half_kelly * ret
        if capital > max_capital:
            max_capital = capital
        dd = (max_capital - capital) / max_capital
        if dd > max_dd:
            max_dd = dd
        trades.append(ret)
        position_end = m15.index[min(fi, len(m15) - 1)]

    t = np.array(trades) if trades else np.array([0])
    return {
        'trades'    : len(t),
        'wr'        : (t > 0).mean() * 100,
        'return'    : (capital / 1000 - 1) * 100,
        'max_dd'    : max_dd * 100,
        'avg_pnl'   : t.mean() * 100,
    }


def walk_forward(df, features, device, n_epochs=100, batch_size=512, conf_threshold=0.70):
    windows = [
        ('2023-01-01', '2024-01-01', '2024-01-01', '2024-07-01'),
        ('2023-01-01', '2024-07-01', '2024-07-01', '2025-01-01'),
        ('2023-01-01', '2025-01-01', '2025-01-01', '2025-07-01'),
        ('2023-01-01', '2025-07-01', '2025-07-01', '2026-04-07'),
    ]

    m15 = pd.read_csv('data/btcusdt_m15.csv', index_col='open_time', parse_dates=True)
    m15 = m15[~m15.index.duplicated(keep='last')].sort_index()

    all_trades = []
    Path('model').mkdir(exist_ok=True)

    for fold, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        print(f"\n{'='*55}")
        print(f"Fold {fold+1}: Train {tr_s[:7]}→{tr_e[:7]} | Test {te_s[:7]}→{te_e[:7]}")

        train_df = df[(df.index >= tr_s) & (df.index < tr_e)]
        test_df  = df[(df.index >= te_s) & (df.index < te_e)]

        val_size  = int(len(train_df) * 0.2)
        val_df    = train_df.iloc[-val_size:]
        train_df2 = train_df.iloc[:-val_size]

        print(f"  Train:{len(train_df2):,} | Val:{len(val_df):,} | Test:{len(test_df):,}")

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train_df2[features]).astype(np.float32)
        X_val   = scaler.transform(val_df[features]).astype(np.float32)
        X_test  = scaler.transform(test_df[features]).astype(np.float32)

        y_train = train_df2['label_bin'].values.astype(np.float32)
        y_val   = val_df['label_bin'].values.astype(np.float32)
        y_test  = test_df['label_bin'].values.astype(np.float32)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=batch_size
        )

        model     = TradingMLP(input_dim=len(features)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCEWithLogitsLoss()
        stopper   = EarlyStopping(patience=12)

        print(f"  {'Ep':>4} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'F1':>6}")
        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_loss, vl_acc, vl_f1 = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(vl_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:>4} {tr_loss:>8.4f} {tr_acc:>7.3f} {vl_loss:>8.4f} {vl_acc:>7.3f} {vl_f1:>6.3f}")

            if stopper.step(vl_loss, model):
                print(f"  Early stop ep={epoch}, best_val={stopper.best_loss:.4f}")
                break

        stopper.restore(model)

        # Inference — probability
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test).to(device))
            proba  = torch.sigmoid(logits).cpu().numpy()

        # Accuracy (conf>0.5)
        preds = (proba > 0.5).astype(float)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, zero_division=0)
        print(f"\n  Accuracy: {acc*100:.1f}%  F1: {f1:.3f}")

        # Confidence threshold testlari
        print(f"  {'Conf':>6} {'Trades':>7} {'WR':>7} {'Return':>8} {'MaxDD':>7}")
        for thr in [0.55, 0.60, 0.65, 0.70]:
            r = backtest(test_df, proba, features, m15, conf_threshold=thr)
            print(f"  {thr:>6.2f} {r['trades']:>7} {r['wr']:>6.1f}% {r['return']:>7.1f}% {r['max_dd']:>6.1f}%")

        # Asosiy natija conf>0.70
        r = backtest(test_df, proba, features, m15, conf_threshold=conf_threshold)
        all_trades.extend([r['avg_pnl']] * r['trades'])

        # Model saqlash
        torch.save({
            'model_state'   : model.state_dict(),
            'scaler_mean'   : scaler.mean_,
            'scaler_std'    : scaler.scale_,
            'features'      : features,
            'input_dim'     : len(features),
            'conf_threshold': conf_threshold,
            'half_kelly'    : 0.081,
            'fold'          : fold + 1,
            'test_period'   : f"{te_s[:7]}→{te_e[:7]}",
        }, f'model/mlp_fold{fold+1}.pt')
        print(f"  Saqlandi: model/mlp_fold{fold+1}.pt")

    all_t = np.array(all_trades)
    print(f"\n{'='*55}")
    print(f"JAMI WALK-FORWARD (conf>{conf_threshold}):")
    print(f"  Avg PnL/trade : {all_t.mean():.3f}%")


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
    walk_forward(df, features, device, n_epochs=100, batch_size=512, conf_threshold=0.70)
