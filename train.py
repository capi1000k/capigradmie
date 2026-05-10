"""
train.py
========
LightGBM — Purged Walk-Forward Cross Validation.

Lopez de Prado metodologiyasi:
    1. Walk-forward split (expanding window)
    2. PURGE  — train/val chegarasida label horizon qadar barlar o'chiriladi
    3. EMBARGO — val dan keyin ham horizon qadar barlar keyingi fold ga kirmaydi

Bu lookahead bias ni butunlay yo'q qiladi.

Embargo = max_bars_max = 16 (Triple Barrier label horizon)

Ishga tushirish:
    python3 train.py
"""

import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

DATASET_FILE = Path("data/dataset.parquet")
MODEL_FILE   = Path("models/lgbm_model.pkl")

N_FOLDS      = 5
TEST_RATIO   = 0.15        # oxirgi 15%
EMBARGO_BARS = 16          # Triple Barrier max_bars_max bilan mos

LGBM_PARAMS = dict(
    objective             = "multiclass",
    num_class             = 3,
    metric                = "multi_logloss",
    learning_rate         = 0.03,
    num_leaves            = 63,
    max_depth             = -1,
    min_child_samples     = 100,
    feature_fraction      = 0.7,
    bagging_fraction      = 0.8,
    bagging_freq          = 5,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    n_estimators          = 3000,
    early_stopping_rounds = 100,
    verbose               = -1,
    n_jobs                = -1,
    random_state          = 42,
)

LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}
LABEL_NAMES   = {-1: "Short", 0: "No-edge", 1: "Long"}


# ──────────────────────────────────────────────
# 1. DATA YUKLASH
# ──────────────────────────────────────────────

def load_dataset():
    print("── Dataset yuklanmoqda ──")
    df = pd.read_parquet(DATASET_FILE)

    num_df     = df.select_dtypes(include=np.number)
    const_cols = num_df.columns[num_df.std() < 1e-10].tolist()
    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        print(f"  Konstant olib tashlandi: {const_cols}")

    X = df.drop(columns=["label"])
    y = df["label"].map(LABEL_MAP).astype(int)

    print(f"  Shape  : {X.shape}")
    print(f"  Davr   : {df.index[0].date()} → {df.index[-1].date()}")

    vc = df["label"].value_counts().sort_index()
    for lbl, cnt in vc.items():
        print(f"  {LABEL_NAMES[lbl]:8s}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    return X, y, df.index


# ──────────────────────────────────────────────
# 2. PURGED WALK-FORWARD SPLIT
# ──────────────────────────────────────────────

def purged_walk_forward_splits(n, n_folds=5, test_ratio=0.15, embargo=16):
    """
    Purged Walk-Forward CV.

    Vizual:
        Fold 1: [===TRAIN===][emb][=VAL=][emb]
        Fold 2: [=====TRAIN=====][emb][=VAL=][emb]
        ...
        TEST:   [emb][====TEST====]

    embargo = label horizon → label overlap yo'q
    """
    test_size  = int(n * test_ratio)
    train_end  = n - test_size - embargo
    fold_size  = train_end // (n_folds + 1)

    splits = []
    for fold in range(n_folds):
        val_start      = fold_size * (fold + 1) + embargo
        val_end        = fold_size * (fold + 2)
        train_end_fold = val_start - embargo

        train_idx = np.arange(0, train_end_fold)
        val_idx   = np.arange(val_start, min(val_end, train_end))

        if len(train_idx) < 500 or len(val_idx) < 500:
            continue

        splits.append((train_idx, val_idx))

    test_start = n - test_size
    test_idx   = np.arange(test_start, n)

    return splits, test_idx


# ──────────────────────────────────────────────
# 3. BITTA FOLD O'QITISH
# ──────────────────────────────────────────────

def train_fold(X_train, y_train, X_val, y_val, fold_num):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    params = {k: v for k, v in LGBM_PARAMS.items()
              if k not in ("n_estimators", "early_stopping_rounds")}

    model = lgb.train(
        params,
        dtrain,
        num_boost_round       = LGBM_PARAMS["n_estimators"],
        valid_sets            = [dval],
        callbacks             = [
            lgb.early_stopping(LGBM_PARAMS["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_proba = model.predict(X_val)
    val_pred  = val_proba.argmax(axis=1)
    val_acc   = (val_pred == y_val.values).mean()

    # Signal precision @ 0.55
    max_p    = val_proba.max(axis=1)
    sig_mask = (max_p >= 0.55) & (val_pred != 1)
    sig_prec = (val_pred[sig_mask] == y_val.values[sig_mask]).mean() if sig_mask.sum() > 0 else 0

    print(f"  Fold {fold_num}: iter={model.best_iteration:>4d}  "
          f"val_acc={val_acc:.4f}  sig_prec@0.55={sig_prec:.4f}  "
          f"train={len(X_train):,}  val={len(X_val):,}")

    return model, val_proba, model.best_iteration


# ──────────────────────────────────────────────
# 4. THRESHOLD TUNING
# ──────────────────────────────────────────────

def tune_threshold(oof_proba, y_true):
    thresholds = np.arange(0.40, 0.80, 0.05)
    pred_class = oof_proba.argmax(axis=1)
    max_proba  = oof_proba.max(axis=1)

    print("\n── Threshold tuning (OOF) ──")
    print(f"  {'Thresh':>7}  {'Signal%':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")

    best_thresh = 0.50
    best_f1     = 0.0

    for thr in thresholds:
        mask     = max_proba >= thr
        sig_pred = pred_class[mask & (pred_class != 1)]
        sig_true = y_true[mask & (pred_class != 1)]

        if len(sig_pred) < 100:
            continue

        precision  = (sig_pred == sig_true).mean()
        true_sig   = (y_true != 1).sum()
        recall     = len(sig_pred) / true_sig if true_sig > 0 else 0
        f1         = 2 * precision * recall / (precision + recall + 1e-9)
        signal_pct = mask.mean() * 100

        print(f"  {thr:>7.2f}  {signal_pct:>7.1f}%  "
              f"{precision:>10.4f}  {recall:>8.4f}  {f1:>6.4f}")

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thr

    print(f"\n  ✅ Eng yaxshi threshold: {best_thresh:.2f}  (F1={best_f1:.4f})")
    return best_thresh


# ──────────────────────────────────────────────
# 5. BAHOLASH
# ──────────────────────────────────────────────

def evaluate(y_true, proba, threshold, label=""):
    pred_raw  = proba.argmax(axis=1)
    max_proba = proba.max(axis=1)
    pred_filt = np.where(max_proba >= threshold, pred_raw, 1)

    print(f"\n── {label} Baholash (threshold={threshold:.2f}) ──")

    sig_mask = pred_filt != 1
    n_sig    = sig_mask.sum()
    print(f"  Signal barlar   : {n_sig:,}  ({n_sig/len(pred_filt)*100:.1f}%)")

    if n_sig > 0:
        prec = (pred_filt[sig_mask] == y_true[sig_mask]).mean()
        print(f"  Signal precision: {prec:.4f}")

    print(f"\n  Classification report:")
    print(classification_report(
        y_true, pred_raw,
        target_names=["Short(-1)", "No-edge(0)", "Long(+1)"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, pred_raw)
    print(f"  Confusion matrix:")
    print(f"  {'':12} Short  No-edge  Long")
    for i, name in enumerate(["Short  ", "No-edge", "Long   "]):
        print(f"  {name}  {cm[i][0]:>6}  {cm[i][1]:>7}  {cm[i][2]:>6}")


# ──────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def show_feature_importance(models, feature_names, top_n=25):
    print(f"\n── Top {top_n} Feature Importance ──")

    importances = np.zeros(len(feature_names))
    for m in models:
        importances += m.feature_importance(importance_type="gain")
    importances /= len(models)

    idx = np.argsort(importances)[::-1][:top_n]
    for rank, i in enumerate(idx, 1):
        bar = "█" * int(importances[i] / importances[idx[0]] * 30)
        print(f"  {rank:>3}. {feature_names[i]:<35} {importances[i]:>10.1f}  {bar}")

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ──────────────────────────────────────────────
# 7. MASTER PIPELINE
# ──────────────────────────────────────────────

def train():
    total_start = time.time()

    print("=" * 60)
    print("  CAPIGRADMIE — Purged Walk-Forward CV")
    print(f"  Embargo: {EMBARGO_BARS} bar  |  Folds: {N_FOLDS}")
    print("=" * 60)

    X, y, index = load_dataset()
    n = len(X)

    splits, test_idx = purged_walk_forward_splits(
        n, N_FOLDS, TEST_RATIO, EMBARGO_BARS
    )

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"\n── Split ──")
    print(f"  Folds    : {len(splits)}")
    print(f"  Test     : {len(test_idx):,} bar  "
          f"({index[test_idx[0]].date()} → {index[test_idx[-1]].date()})")
    print(f"  Embargo  : {EMBARGO_BARS} bar har fold orasida")

    # Walk-forward CV
    print(f"\n── Purged Walk-Forward CV ──")

    models     = []
    oof_probas = []
    oof_trues  = []
    best_iters = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        model, val_proba, best_iter = train_fold(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx],   y.iloc[val_idx],
            fold_num,
        )
        oof_probas.append(val_proba)
        oof_trues.extend(y.iloc[val_idx].tolist())
        models.append(model)
        best_iters.append(best_iter)

    oof_proba = np.vstack(oof_probas)
    oof_true  = np.array(oof_trues)

    # OOF baholash
    evaluate(oof_true, oof_proba, 0.50, label="OOF")

    # Threshold tuning
    threshold = tune_threshold(oof_proba, oof_true)

    # Final model
    print(f"\n── Final model ──")
    avg_iter   = int(np.mean(best_iters))
    print(f"  O'rtacha best iter: {avg_iter}")

    train_end  = test_idx[0] - EMBARGO_BARS
    X_trainval = X.iloc[:train_end]
    y_trainval = y.iloc[:train_end]

    params_final = {k: v for k, v in LGBM_PARAMS.items()
                    if k not in ("n_estimators", "early_stopping_rounds")}

    final_model = lgb.train(
        params_final,
        lgb.Dataset(X_trainval, label=y_trainval),
        num_boost_round = avg_iter,
        callbacks       = [lgb.log_evaluation(period=-1)],
    )
    print(f"  ✅ Final model tayyor  ({len(X_trainval):,} bar)")

    # Test baholash
    test_proba = final_model.predict(X_test)
    evaluate(y_test.values, test_proba, threshold, label="TEST SET")

    # Feature importance
    feat_imp = show_feature_importance(
        models + [final_model],
        X.columns.tolist(),
        top_n=25,
    )

    # Saqlash
    Path("models").mkdir(exist_ok=True)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "threshold":     threshold,
            "feature_names": X.columns.tolist(),
            "label_map":     LABEL_MAP,
            "label_map_inv": LABEL_MAP_INV,
            "best_iter":     avg_iter,
            "embargo_bars":  EMBARGO_BARS,
            "lgbm_params":   LGBM_PARAMS,
        }, f)

    feat_imp.to_csv("models/feature_importance.csv")

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  ✅ YAKUNLANDI  —  {elapsed:.1f} daqiqa")
    print(f"  Model: {MODEL_FILE}")
    print(f"{'='*60}\n")

    return final_model, threshold, feat_imp


if __name__ == "__main__":
    model, threshold, feat_imp = train()