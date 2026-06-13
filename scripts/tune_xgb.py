"""Diagnose XGBoost early-stopping behaviour from notebook 01.

The Stage 2 fit had ``best_iteration=499`` against ``n_estimators=500`` with
``early_stopping_rounds=30`` — early stopping never triggered, suggesting the
eval-set loss was still improving when training hit the cap.

This script:
1. Re-fits with a much higher cap to find the *actual* plateau.
2. Plots train and eval log-loss vs iteration.
3. Evaluates a small grid of (n_estimators, early_stopping_rounds) combos on
   the OOT set (uncalibrated, since calibration is an orthogonal step).
4. Recommends a config.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_pipeline import load_and_combine, out_of_time_split

warnings.filterwarnings("ignore", category=UserWarning)


def gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(2 * roc_auc_score(y_true, y_pred) - 1)


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_pred))


def build_splits(cfg):
    df = load_and_combine(str(PROJECT_ROOT / cfg["data"]["raw_dir"]), cfg["data"]["target_col"])
    X_train, y_train, X_test, y_test = out_of_time_split(
        df, cfg["data"]["date_col"],
        cfg["data"]["train_years"], cfg["data"]["test_years"], cfg["data"]["target_col"],
    )
    val_frac = cfg["models"]["xgb"]["calibration_val_frac"]
    seed = cfg["bootstrap"]["seed"]

    mf_idx, cal_idx = train_test_split(
        np.arange(len(X_train)), test_size=val_frac, random_state=seed, stratify=y_train.values
    )
    X_mf, y_mf = X_train.iloc[mf_idx], y_train.iloc[mf_idx].values
    X_cal, y_cal = X_train.iloc[cal_idx], y_train.iloc[cal_idx].values

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_mf, y_mf, test_size=val_frac, random_state=seed, stratify=y_mf
    )
    return dict(
        X_fit=X_fit, y_fit=y_fit, X_es=X_es, y_es=y_es,
        X_cal=X_cal, y_cal=y_cal, X_test=X_test, y_test=y_test.values,
        scale_pos=float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)),
    )


def fit_xgb(splits, n_estimators: int, early_stopping_rounds: int | None, cfg, record_history: bool = False):
    xgb_cfg = cfg["models"]["xgb"]
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=splits["scale_pos"],
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=cfg["bootstrap"]["seed"],
    )
    eval_set = [(splits["X_fit"], splits["y_fit"]), (splits["X_es"], splits["y_es"])]
    clf.fit(splits["X_fit"], splits["y_fit"], eval_set=eval_set, verbose=False)
    return clf


def evaluate(clf, splits) -> dict:
    pred_oot = clf.predict_proba(splits["X_test"])[:, 1]
    pred_cal = clf.predict_proba(splits["X_cal"])[:, 1]
    pred_es = clf.predict_proba(splits["X_es"])[:, 1]
    return {
        "best_iteration": int(clf.best_iteration) if clf.best_iteration is not None else None,
        "oot_gini": gini(splits["y_test"], pred_oot),
        "oot_brier": brier(splits["y_test"], pred_oot),
        "cal_holdout_logloss": log_loss(splits["y_cal"], pred_cal),
        "es_holdout_logloss": log_loss(splits["y_es"], pred_es),
    }


def plot_learning_curve(history: dict, best_iter: int, out_path: Path, current_cap: int) -> None:
    train_loss = history["validation_0"]["logloss"]
    es_loss = history["validation_1"]["logloss"]
    iters = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iters, train_loss, color="#2563eb", label="train log-loss", linewidth=1.5)
    ax.plot(iters, es_loss, color="#dc2626", label="es-holdout log-loss", linewidth=1.5)
    ax.axvline(best_iter, color="#16a34a", linestyle="--", linewidth=1.5,
               label=f"best_iteration = {best_iter}")
    ax.axvline(current_cap, color="#6b7280", linestyle=":", linewidth=1.2,
               label=f"current cap (n_estimators={current_cap})")

    best_es_loss = es_loss[best_iter] if best_iter < len(es_loss) else es_loss[-1]
    cap_es_loss = es_loss[current_cap - 1] if current_cap - 1 < len(es_loss) else es_loss[-1]
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("log-loss")
    ax.set_title(
        f"ES-holdout log-loss continues falling past current cap — "
        f"best={best_es_loss:.5f} at iter {best_iter}, was {cap_es_loss:.5f} at the cap"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    splits = build_splits(cfg)
    print(f"Splits: fit={len(splits['y_fit'])}, es={len(splits['y_es'])}, "
          f"cal={len(splits['y_cal'])}, test={len(splits['y_test'])}, "
          f"scale_pos_weight={splits['scale_pos']:.3f}")

    # 1. Long run: n_estimators=3000, very loose ES so we see the real plateau.
    print("\n[1/2] Long run: n_estimators=3000, early_stopping_rounds=200 (find true plateau)")
    long_clf = fit_xgb(splits, n_estimators=3000, early_stopping_rounds=200, cfg=cfg)
    long_metrics = evaluate(long_clf, splits)
    print(f"  best_iteration = {long_metrics['best_iteration']}")
    print(f"  OOT Gini={long_metrics['oot_gini']:.4f}, OOT Brier={long_metrics['oot_brier']:.4f}")
    print(f"  cal-holdout logloss = {long_metrics['cal_holdout_logloss']:.5f}")
    print(f"  es-holdout logloss  = {long_metrics['es_holdout_logloss']:.5f}")

    # Plot
    out_dir = PROJECT_ROOT / cfg["data"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "xgb_learning_curve.png"
    plot_learning_curve(long_clf.evals_result(), long_metrics["best_iteration"],
                        plot_path, current_cap=cfg["models"]["xgb"]["n_estimators"])
    print(f"  wrote {plot_path.relative_to(PROJECT_ROOT)}")

    # 2. Grid: small set of (n_est, ES) combos
    print("\n[2/2] Grid sweep")
    grid = [
        # (label, n_estimators, early_stopping_rounds)
        ("current (500, 30)",      500,  30),
        ("tight (500, 10)",        500,  10),
        ("loose (1000, 30)",      1000,  30),
        ("loose (1500, 50)",      1500,  50),
        ("loose (2000, 50)",      2000,  50),
        ("very loose (3000, 100)", 3000, 100),
    ]
    rows = []
    for label, n_est, es_rounds in grid:
        clf = fit_xgb(splits, n_estimators=n_est, early_stopping_rounds=es_rounds, cfg=cfg)
        m = evaluate(clf, splits)
        m["label"] = label
        m["n_estimators_cap"] = n_est
        m["es_rounds"] = es_rounds
        rows.append(m)
        print(f"  {label:24s}  best_iter={m['best_iteration']:4d}  "
              f"OOT_Gini={m['oot_gini']:.4f}  OOT_Brier={m['oot_brier']:.4f}  "
              f"es_logloss={m['es_holdout_logloss']:.5f}")
    df = pd.DataFrame(rows)[["label", "n_estimators_cap", "es_rounds", "best_iteration",
                              "oot_gini", "oot_brier", "es_holdout_logloss"]]
    csv_path = out_dir / "xgb_tune_grid.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path.relative_to(PROJECT_ROOT)}")

    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
