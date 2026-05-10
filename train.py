"""
train.py
========
LightGBM model — Walk-Forward Cross Validation.

Pipeline:
    1. dataset.parquet yuklash
    2. Walk-forward 5 fold split
    3. Har fold da LightGBM o'qitish
    4. Out-of-fold (OOF) prediction yig'ish
    5. Threshold tuning — confidence filter
    6. Yakuniy test set baholash
    7. Feature importance
    8. model.pkl saqlash

Ishga tushirish:
    pip install lightgbm scikit-learn pandas pyarrow
    python3 train.py
"""

import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

DATASET_FILE = Path("data/dataset.parquet")
MODEL_FILE   = Path("models/lgbm_model.pkl")
RESULTS_FILE = Path("models/oof_predictions.parquet")

N_FOLDS      = 5
TEST_RATIO   = 0.10      # oxirgi 10% — hech qachon train da emas
VAL_RATIO    = 0.10      # har fold ichida validation

# LightGBM parametrlar
LGBM_PARAMS = dict(
    objective        = "multiclass",
    num_class        = 3,
    metric           = "multi_logloss",
    learning_rate    = 0.05,
    num_leaves       = 64,
    max_depth        = 6,
    min_child_samples= 50,
    feature_fraction = 0.7,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    n_estimators     = 1000,
    early_stopping_rounds = 50,
    verbose          = -1,
    n_jobs           = -1,
    random_state     = 42,
)

# Label mapping: LightGBM 0,1,2 kerak → bizda -1,0,+1
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}
LABEL_NAMES   = {-1: "Short", 0: "No-edge", 1: "Long"}


# ──────────────────────────────────────────────
# 1. DATA YUKLASH
# ──────────────────────────────────────────────

def load_dataset():
    print("── Dataset yuklanmoqda ──")
    df = pd.read_parquet(DATASET_FILE)

    # Konstant ustunlarni olib tashlash
    num_df     = df.select_dtypes(include=np.number)
    const_cols = num_df.columns[num_df.std() < 1e-10].tolist()
    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        print(f"  Konstant ustunlar olib tashlandi: {const_cols}")

    X = df.drop(columns=["label"])
    y = df["label"].map(LABEL_MAP).astype(int)

    print(f"  Shape  : {X.shape}")
    print(f"  Davr   : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Feature: {X.shape[1]}")

    vc = df["label"].value_counts().sort_index()
    for lbl, cnt in vc.items():
        print(f"  {LABEL_NAMES[lbl]:8s}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    return X, y, df.index


# ──────────────────────────────────────────────
# 2. WALK-FORWARD SPLIT
# ──────────────────────────────────────────────

def walk_forward_splits(n: int,
                        n_folds: int = 5,
                        test_ratio: float = 0.10):
    """
    Walk-forward CV split generatori.

    Test set (oxirgi test_ratio) — hech qachon train ga kirmaydi.
    Qolgan qism n_folds ga bo'linadi.

    Fold 1: |--train--|val|
    Fold 2: |----train----|val|
    Fold 3: |------train------|val|
    ...

    Expanding window: har fold da train kengayadi.
    Bu real trading sharoitiga eng yaqin.
    """
    test_size  = int(n * test_ratio)
    train_end  = n - test_size
    fold_size  = train_end // (n_folds + 1)

    splits = []
    for fold in range(n_folds):
        val_end   = fold_size * (fold + 2)
        val_start = fold_size * (fold + 1)
        train_idx = np.arange(0, val_start)
        val_idx   = np.arange(val_start, min(val_end, train_end))
        splits.append((train_idx, val_idx))

    test_idx = np.arange(train_end, n)

    return splits, test_idx


# ──────────────────────────────────────────────
# 3. BITTA FOLD O'QITISH
# ──────────────────────────────────────────────

def train_fold(X_train, y_train,
               X_val, y_val,
               fold_num: int):
    """
    Bitta fold uchun LightGBM o'qitish.
    Early stopping — val loss ga qarab.
    """
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(LGBM_PARAMS["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    params = {k: v for k, v in LGBM_PARAMS.items()
              if k not in ("n_estimators", "early_stopping_rounds")}

    model = lgb.train(
        params,
        dtrain,
        num_boost_round   = LGBM_PARAMS["n_estimators"],
        valid_sets        = [dval],
        callbacks         = callbacks,
    )

    best_iter = model.best_iteration
    val_proba = model.predict(X_val)
    val_pred  = val_proba.argmax(axis=1)
    val_acc   = (val_pred == y_val.values).mean()

    print(f"  Fold {fold_num}: iter={best_iter:>4d}  val_acc={val_acc:.4f}")

    return model, val_proba, best_iter


# ──────────────────────────────────────────────
# 4. THRESHOLD TUNING
# ──────────────────────────────────────────────

def tune_threshold(oof_proba: np.ndarray,
                   y_true: np.ndarray,
                   thresholds: np.ndarray = None) -> float:
    """
    Confidence threshold tanlash.

    Model har bar uchun 3 ta ehtimol beradi:
        [P(short), P(no-edge), P(long)]

    max(proba) > threshold bo'lsagina signal beramiz.
    Aks holda → no-edge (0) deb qabul qilamiz.

    Maqsad: Precision ni maksimallash (yolg'on signal minimal).
    """
    if thresholds is None:
        thresholds = np.arange(0.35, 0.80, 0.05)

    pred_classes = oof_proba.argmax(axis=1)
    max_proba    = oof_proba.max(axis=1)

    print("\n── Threshold tuning ──")
    print(f"  {'Threshold':>10}  {'Signal%':>8}  {'Precision':>10}  {'Recall':>8}")

    best_thresh  = 0.35
    best_f1      = 0.0

    for thr in thresholds:
        mask      = max_proba >= thr
        pred_filt = np.where(mask, pred_classes, 1)  # 1 = no-edge

        # Faqat signal barlar uchun precision
        signal_mask = (pred_filt != 1) & (y_true != 1)
        if signal_mask.sum() < 100:
            continue

        correct    = (pred_filt[signal_mask] == y_true[signal_mask]).sum()
        precision  = correct / signal_mask.sum()
        signal_pct = mask.mean() * 100

        # Recall: haqiqiy signallarning qanchasi ushlandi
        true_signal = (y_true != 1).sum()
        recall = signal_mask.sum() / true_signal if true_signal > 0 else 0

        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"  {thr:>10.2f}  {signal_pct:>7.1f}%  {precision:>10.4f}  {recall:>8.4f}")

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thr

    print(f"\n  ✅ Eng yaxshi threshold: {best_thresh:.2f}  (F1={best_f1:.4f})")
    return best_thresh


# ──────────────────────────────────────────────
# 5. BAHOLASH
# ──────────────────────────────────────────────

def evaluate(y_true: np.ndarray,
             proba: np.ndarray,
             threshold: float,
             label: str = ""):
    """
    Model sifatini baholash.
    Threshold qo'llangan va qo'llanmagan holda.
    """
    pred_raw  = proba.argmax(axis=1)
    max_proba = proba.max(axis=1)
    pred_filt = np.where(max_proba >= threshold, pred_raw, 1)

    print(f"\n── {label} Baholash (threshold={threshold:.2f}) ──")

    # Signal filtrlangandan keyin
    signal_mask  = pred_filt != 1
    signal_count = signal_mask.sum()
    signal_pct   = signal_count / len(pred_filt) * 100

    print(f"  Signal barlar: {signal_count:,}  ({signal_pct:.1f}%)")

    if signal_count > 0:
        correct   = (pred_filt[signal_mask] == y_true[signal_mask]).sum()
        precision = correct / signal_count
        print(f"  Signal precision: {precision:.4f}")

    # Classification report (raw, threshold yo'q)
    target_names = ["Short(-1)", "No-edge(0)", "Long(+1)"]
    print(f"\n  Raw classification report:")
    print(classification_report(
        y_true, pred_raw,
        target_names=target_names,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, pred_raw)
    print(f"  Confusion matrix:")
    print(f"  {'':12} Short  No-edge  Long")
    for i, row_name in enumerate(["Short  ", "No-edge", "Long   "]):
        print(f"  {row_name}  {cm[i][0]:>6}  {cm[i][1]:>7}  {cm[i][2]:>6}")


# ──────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def show_feature_importance(models: list,
                             feature_names: list,
                             top_n: int = 20):
    """
    Barcha fold modellarning o'rtacha feature importance.
    Top N featurelarni ko'rsatadi.
    """
    print(f"\n── Top {top_n} Feature Importance ──")

    importances = np.zeros(len(feature_names))
    for model in models:
        importances += model.feature_importance(importance_type="gain")

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

    print("=" * 55)
    print("  CAPIGRADMIE — LightGBM Trainer")
    print("=" * 55)

    # 1. Data
    X, y, index = load_dataset()
    n = len(X)

    # 2. Split
    splits, test_idx = walk_forward_splits(n, N_FOLDS, TEST_RATIO)

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"\n── Walk-Forward Split ──")
    print(f"  Train+Val: {len(X) - len(test_idx):,} bar")
    print(f"  Test     : {len(test_idx):,} bar  "
          f"({index[test_idx[0]].date()} → {index[test_idx[-1]].date()})")

    # 3. Walk-forward CV
    print(f"\n── {N_FOLDS} Fold Walk-Forward CV ──")

    models       = []
    oof_proba    = np.zeros((n - len(test_idx), 3))
    oof_true     = []
    best_iters   = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_vl = X.iloc[val_idx]
        y_vl = y.iloc[val_idx]

        t0 = time.time()
        model, val_proba, best_iter = train_fold(X_tr, y_tr, X_vl, y_vl, fold_num)

        # OOF yig'ish
        oof_proba[val_idx] = val_proba
        oof_true.extend(y_vl.tolist())

        models.append(model)
        best_iters.append(best_iter)

    oof_true = np.array(oof_true)

    # 4. Threshold tuning (OOF da)
    # Faqat val indekslari bo'lgan qatorlar
    all_val_idx = np.concatenate([val_idx for _, val_idx in splits])
    oof_proba_valid = oof_proba[all_val_idx]

    threshold = tune_threshold(oof_proba_valid, oof_true)

    # 5. Final model — barcha train+val datada
    print(f"\n── Final model o'qitilmoqda ──")
    avg_best_iter = int(np.mean(best_iters))
    print(f"  O'rtacha best iteration: {avg_best_iter}")

    train_val_end = n - len(test_idx)
    X_trainval = X.iloc[:train_val_end]
    y_trainval = y.iloc[:train_val_end]

    params_final = {k: v for k, v in LGBM_PARAMS.items()
                    if k not in ("n_estimators", "early_stopping_rounds")}

    final_model = lgb.train(
        params_final,
        lgb.Dataset(X_trainval, label=y_trainval),
        num_boost_round = avg_best_iter,
        callbacks       = [lgb.log_evaluation(period=-1)],
    )
    print(f"  ✅ Final model tayyor")

    # 6. Test baholash
    test_proba = final_model.predict(X_test)
    evaluate(y_test.values, test_proba, threshold, label="TEST SET")

    # 7. Feature importance
    feat_imp = show_feature_importance(
        models + [final_model],
        X.columns.tolist(),
        top_n=20
    )

    # 8. Saqlash
    Path("models").mkdir(exist_ok=True)

    save_obj = {
        "model":          final_model,
        "threshold":      threshold,
        "feature_names":  X.columns.tolist(),
        "label_map":      LABEL_MAP,
        "label_map_inv":  LABEL_MAP_INV,
        "best_iter":      avg_best_iter,
        "lgbm_params":    LGBM_PARAMS,
        "target_params":  {
            "atr_mult_tp": 2.5, "atr_mult_sl": 2.5,
            "max_bars_min": 4,  "max_bars_max": 16,
        },
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_obj, f)

    feat_imp.to_csv("models/feature_importance.csv")

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*55}")
    print(f"  ✅ O'QITISH YAKUNLANDI  —  {elapsed:.1f} daqiqa")
    print(f"  Model: {MODEL_FILE}")
    print(f"{'='*55}\n")

    return final_model, threshold, feat_imp


# ──────────────────────────────────────────────
# ISHGA TUSHIRISH
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model, threshold, feat_imp = train()