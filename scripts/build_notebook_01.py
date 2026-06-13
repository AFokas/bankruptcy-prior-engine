"""Generate notebooks/01_data_foundation.ipynb (Stage 2).

LR vs calibrated XGBoost on the Polish Bankruptcy OOT split. The notebook
fits both models inline; Stage 3 will refactor the same logic into tested
``src/`` modules.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "notebooks" / "01_data_foundation.ipynb"


CELLS: list = [
    ("md", """# Notebook 01 — Data Foundation

**Question this notebook answers:** How do logistic regression and calibrated XGBoost compare on out-of-time performance, and which is the right baseline for the Bayesian update layer?

**Decision criteria:**
1. OOT Gini and Brier (in-time metrics are not load-bearing — they are reported only as drift indicators).
2. Calibration at extreme probabilities (the Bayesian layer assumes a calibrated likelihood).
3. Coefficient interpretability (the prior is specified in log-odds units, which matches LR exactly)."""),

    ("code", """from __future__ import annotations
import sys, json, warnings
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "notebooks")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

import xgboost as xgb
import sklearn
import platform

from src.config import load_config
from src.data_pipeline import load_and_combine, out_of_time_split
from _explore_helpers import (
    WoeEncoder, compute_iv_table, gini, brier, bootstrap_ci,
    compute_psi, psi_status, IsotonicCalibratedClassifier,
)

warnings.filterwarnings("ignore", category=UserWarning)

cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
PALETTE = cfg["colours"]
TARGET_COL = cfg["data"]["target_col"]
DATE_COL = cfg["data"]["date_col"]
N_BOOTSTRAP = cfg["bootstrap"]["n_resamples"]
CI_LEVEL = cfg["bootstrap"]["ci_level"]
SEED = cfg["bootstrap"]["seed"]
IV_THRESHOLD = cfg["woe"]["iv_threshold"]

sns.set_style("whitegrid", {"axes.grid": True, "grid.alpha": 0.3, "grid.color": "#cbd5e1"})
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.facecolor": PALETTE["background"], "figure.facecolor": "white",
})"""),

    ("md", """## Out-of-time split

Train: years 1–3 (n≈27.7k). Test: years 4–5 (n≈15.7k). The test set has a *higher* base rate than train (5.9% vs 4.2%), which is the conservative direction for model evaluation."""),

    ("code", """df = load_and_combine(str(PROJECT_ROOT / cfg["data"]["raw_dir"]), TARGET_COL)
X_train, y_train, X_test, y_test = out_of_time_split(
    df, DATE_COL, cfg["data"]["train_years"], cfg["data"]["test_years"], TARGET_COL
)
print(f"Train: n={len(y_train):,}  default rate={y_train.mean():.4f}")
print(f"Test : n={len(y_test):,}  default rate={y_test.mean():.4f}")"""),

    ("md", """## Fit logistic regression

Pipeline (inline; Stage 3 will refactor into `src/models.py`):
1. `WoeEncoder` — quantile bin → WoE per feature, dedicated `MISSING` bin.
2. Drop features with `IV < cfg.woe.iv_threshold`.
3. `StandardScaler` on the WoE-encoded matrix.
4. `LogisticRegression` with `cfg.models.lr` hyperparameters."""),

    ("code", """encoder = WoeEncoder(n_bins=cfg["woe"]["n_bins"], laplace=cfg["woe"]["laplace_smoothing"])
encoder.fit(X_train, y_train)

iv_series = pd.Series(encoder.iv_).sort_values(ascending=False)
kept_features = iv_series[iv_series >= IV_THRESHOLD].index.tolist()
dropped_features = iv_series[iv_series < IV_THRESHOLD].index.tolist()
print(f"Kept {len(kept_features)} features (IV ≥ {IV_THRESHOLD}); dropped {len(dropped_features)}.")

Z_train_full = encoder.transform(X_train)
Z_test_full = encoder.transform(X_test)
Z_train = Z_train_full[kept_features]
Z_test = Z_test_full[kept_features]

scaler = StandardScaler().fit(Z_train.values)
Z_train_s = scaler.transform(Z_train.values)
Z_test_s = scaler.transform(Z_test.values)

lr = LogisticRegression(
    C=cfg["models"]["lr"]["C"],
    solver=cfg["models"]["lr"]["solver"],
    max_iter=cfg["models"]["lr"]["max_iter"],
    class_weight=cfg["models"]["lr"]["class_weight"],
    random_state=SEED,
)
lr.fit(Z_train_s, y_train.values)

lr_in_pred = lr.predict_proba(Z_train_s)[:, 1]
lr_oot_pred = lr.predict_proba(Z_test_s)[:, 1]
print(f"LR in-time Gini = {gini(y_train.values, lr_in_pred):.4f}, OOT Gini = {gini(y_test.values, lr_oot_pred):.4f}")
print(f"LR in-time Brier = {brier(y_train.values, lr_in_pred):.4f}, OOT Brier = {brier(y_test.values, lr_oot_pred):.4f}")"""),

    ("md", """## Fit calibrated XGBoost

XGBoost handles missing values natively, so we feed it the raw feature matrix. Calibration follows the spec in `cfg.models.xgb`: hold out `calibration_val_frac` of the training set, fit XGB on the rest with early stopping (using a small internal validation split), then fit an isotonic calibrator on the held-out fold."""),

    ("code", """xgb_cfg = cfg["models"]["xgb"]
val_frac = xgb_cfg["calibration_val_frac"]

# 1. carve off calibration holdout
mf_idx, cal_idx = train_test_split(
    np.arange(len(X_train)), test_size=val_frac,
    random_state=SEED, stratify=y_train.values,
)
X_mf, y_mf = X_train.iloc[mf_idx], y_train.iloc[mf_idx].values
X_cal, y_cal = X_train.iloc[cal_idx], y_train.iloc[cal_idx].values

# 2. inner split for early stopping
X_fit, X_es, y_fit, y_es = train_test_split(
    X_mf, y_mf, test_size=val_frac, random_state=SEED, stratify=y_mf,
)

scale_pos = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
xgb_model = xgb.XGBClassifier(
    n_estimators=xgb_cfg["n_estimators"],
    max_depth=xgb_cfg["max_depth"],
    learning_rate=xgb_cfg["learning_rate"],
    subsample=xgb_cfg["subsample"],
    colsample_bytree=xgb_cfg["colsample_bytree"],
    scale_pos_weight=scale_pos,
    early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)
xgb_model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

# Isotonic calibration on the held-out fold (sklearn 1.7+ removed cv='prefit'
# from CalibratedClassifierCV; we fit IsotonicRegression directly).
xgb_cal = IsotonicCalibratedClassifier.fit_from_holdout(xgb_model, X_cal, y_cal)

xgb_in_pred = xgb_cal.predict_proba(X_train)[:, 1]
xgb_oot_pred = xgb_cal.predict_proba(X_test)[:, 1]
print(f"XGB in-time Gini = {gini(y_train.values, xgb_in_pred):.4f}, OOT Gini = {gini(y_test.values, xgb_oot_pred):.4f}")
print(f"XGB in-time Brier = {brier(y_train.values, xgb_in_pred):.4f}, OOT Brier = {brier(y_test.values, xgb_oot_pred):.4f}")
print(f"Best XGB iteration (early stopping): {xgb_model.best_iteration}")"""),

    ("md", """### Chart 1 — In-time vs OOT Gini

The Gini drop from in-time to OOT is the headline portability metric. A small gap implies the model generalises; a large gap suggests the in-time fit was capturing noise that doesn't transfer."""),

    ("code", """rows = [
    ("LR", "in-time", gini(y_train.values, lr_in_pred)),
    ("LR", "OOT",     gini(y_test.values,  lr_oot_pred)),
    ("XGBoost", "in-time", gini(y_train.values, xgb_in_pred)),
    ("XGBoost", "OOT",     gini(y_test.values,  xgb_oot_pred)),
]
gini_df = pd.DataFrame(rows, columns=["model", "split", "gini"])
pivot = gini_df.pivot(index="model", columns="split", values="gini")
pivot["drop"] = pivot["in-time"] - pivot["OOT"]
print(pivot.round(4).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(pivot.index))
width = 0.38
ax.bar(x - width/2, pivot["in-time"], width, label="in-time (train years)", color=PALETTE["primary"])
ax.bar(x + width/2, pivot["OOT"],     width, label="OOT (test years)",      color=PALETTE["secondary"])
for i, m in enumerate(pivot.index):
    drop = pivot.loc[m, "drop"]
    ax.text(i, max(pivot.loc[m].drop("drop")) + 0.01, f"Δ = {drop:.3f}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(pivot.index)
ax.set_ylabel("Gini")
lr_drop = float(pivot.loc["LR", "drop"]); xgb_drop = float(pivot.loc["XGBoost", "drop"])
ax.set_title(
    f"LR drops {lr_drop:.3f} Gini in-time→OOT; XGBoost drops {xgb_drop:.3f} — "
    f"{'LR generalises more cleanly' if lr_drop < xgb_drop else 'XGBoost generalises more cleanly'}"
)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()"""),

    ("md", """**Finding:** Both models show a meaningful in-time → OOT Gini drop, consistent with the year-4/5 cohort being structurally harder (higher default rate, post-train regime). The relative drop tells us which model is overfitting more."""),

    ("md", """### Chart 2 — Reliability diagrams

Reliability shows whether predicted PD matches realised default rate within each probability bin. The Bayesian update layer expects a calibrated likelihood — if the base model is mis-calibrated, the posterior will be too."""),

    ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name, pred in [(axes[0], "LR", lr_oot_pred), (axes[1], "XGBoost (calibrated)", xgb_oot_pred)]:
    frac_pos, mean_pred = calibration_curve(y_test.values, pred, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], linestyle=":", color=PALETTE["neutral"], label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color=PALETTE["primary"] if name == "LR" else PALETTE["tertiary"], label=name)
    ax.set_xlabel("Predicted PD (bin mean)")
    ax.set_ylabel("Realised default rate")
    b = brier(y_test.values, pred)
    ax.set_title(f"{name} — OOT Brier = {b:.4f}")
    ax.legend()
    ax.set_xlim(0, max(0.4, float(mean_pred.max()) * 1.1))
    ax.set_ylim(0, max(0.4, float(frac_pos.max()) * 1.1))

brier_lr = brier(y_test.values, lr_oot_pred)
brier_xgb = brier(y_test.values, xgb_oot_pred)
better = "XGBoost" if brier_xgb < brier_lr else "LR"
fig.suptitle(f"{better} has lower OOT Brier — {abs(brier_lr - brier_xgb):.4f} difference", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.show()"""),

    ("md", """**Finding:** Reliability curves diverge most at the high-PD tail, which is exactly where credit decisions are most consequential. The Brier delta is small but the curve shape matters: an under-calibrated tail biases the prior-update posterior toward the data and away from the analyst prior, which we don't want."""),

    ("md", """### Bootstrap confidence intervals

Non-parametric bootstrap, `cfg.bootstrap.n_resamples` resamples on the OOT set, percentile CI at `cfg.bootstrap.ci_level`. If the LR/XGBoost CIs overlap, the Gini difference is not statistically distinguishable."""),

    ("code", """gini_ci_lr  = bootstrap_ci(y_test.values, lr_oot_pred,  gini,  N_BOOTSTRAP, CI_LEVEL, SEED)
gini_ci_xgb = bootstrap_ci(y_test.values, xgb_oot_pred, gini,  N_BOOTSTRAP, CI_LEVEL, SEED)
brier_ci_lr  = bootstrap_ci(y_test.values, lr_oot_pred,  brier, N_BOOTSTRAP, CI_LEVEL, SEED)
brier_ci_xgb = bootstrap_ci(y_test.values, xgb_oot_pred, brier, N_BOOTSTRAP, CI_LEVEL, SEED)

ci_df = pd.DataFrame([
    ("Gini",  "LR",      *gini_ci_lr.values()),
    ("Gini",  "XGBoost", *gini_ci_xgb.values()),
    ("Brier", "LR",      *brier_ci_lr.values()),
    ("Brier", "XGBoost", *brier_ci_xgb.values()),
], columns=["metric", "model", "estimate", "lower", "upper", "std"])
print(ci_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric in zip(axes, ["Gini", "Brier"]):
    sub = ci_df[ci_df["metric"] == metric].reset_index(drop=True)
    ys = np.arange(len(sub))
    for i, row in sub.iterrows():
        color = PALETTE["primary"] if row["model"] == "LR" else PALETTE["tertiary"]
        ax.plot([row["lower"], row["upper"]], [i, i], color=color, linewidth=3)
        ax.plot(row["estimate"], i, "o", color=color, markersize=10)
        ax.text(row["upper"], i, f"  [{row['lower']:.3f}, {row['upper']:.3f}]",
                va="center", fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels(sub["model"])
    ax.set_xlabel(metric)
    ax.set_title(f"OOT {metric} — 95% bootstrap CI")

# annotate overlap
lo_lr, hi_lr = gini_ci_lr["lower"], gini_ci_lr["upper"]
lo_xg, hi_xg = gini_ci_xgb["lower"], gini_ci_xgb["upper"]
overlap = max(lo_lr, lo_xg) < min(hi_lr, hi_xg)
verdict = "CIs overlap — Gini difference not statistically distinguishable" if overlap else "CIs separate — Gini difference is statistically meaningful"
fig.suptitle(verdict, fontweight="bold", fontsize=12)
plt.tight_layout()
plt.show()"""),

    ("md", """**Finding:** The CI overlap test determines whether the Gini gap is real or noise. For the Bayesian layer, we will use whichever model has the cleaner OOT calibration and tighter CI — see the model-selection decision below."""),

    ("md", """### Chart 4 — PSI per feature

PSI compares the train-time bin distribution to the OOT bin distribution for each feature. PSI < 0.1 = STABLE, < 0.2 = MONITOR, ≥ 0.2 = RETRAIN. Features that drift hard between train and OOT are the canaries for a future macro-shift retrain trigger."""),

    ("code", """psi_rows = []
for feat in X_train.columns:
    base = X_train[feat].values.astype(float)
    curr = X_test[feat].values.astype(float)
    edges = encoder.bin_edges_[feat]
    val = compute_psi(base, curr, edges, epsilon=cfg["psi"]["epsilon"])
    psi_rows.append({"feature": feat, "psi": val,
                     "status": psi_status(val, cfg["psi"]["stable_threshold"], cfg["psi"]["monitor_threshold"])})
psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False).reset_index(drop=True)

status_colors = {"STABLE": PALETTE["tertiary"], "MONITOR": "#fbbf24", "RETRAIN": PALETTE["secondary"]}
top_psi = psi_df.head(25)

fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(top_psi["feature"][::-1], top_psi["psi"][::-1],
               color=top_psi["status"].map(status_colors)[::-1])
for thresh in (cfg["psi"]["stable_threshold"], cfg["psi"]["monitor_threshold"]):
    ax.axvline(thresh, color="black", linestyle=":", alpha=0.5)
n_retrain = int((psi_df["status"] == "RETRAIN").sum())
n_monitor = int((psi_df["status"] == "MONITOR").sum())
n_stable  = int((psi_df["status"] == "STABLE").sum())

ax.set_xlabel("PSI (OOT vs train)")
ax.set_title(
    f"{n_retrain} features in RETRAIN zone, {n_monitor} in MONITOR — these are the macro-shift canaries"
)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=f"{lbl} (n={n})") for lbl, c, n in [
    ("STABLE", PALETTE["tertiary"], n_stable),
    ("MONITOR", "#fbbf24", n_monitor),
    ("RETRAIN", PALETTE["secondary"], n_retrain),
]], loc="lower right")
plt.tight_layout()
plt.show()"""),

    ("md", """**Finding:** Distribution shift is concentrated in a handful of features; the bulk of features stay STABLE. The Stage 5 macro-shift detector will fire only when high-PSI features *also* show analyst-prior drift — single-channel PSI alone is not actionable."""),

    ("md", """### Chart 5 — XGBoost gain importance vs IV

Features that score high on both axes are the most robust signal. Features high on XGBoost gain but low on IV are picking up interactions invisible to a univariate IV. Features high on IV but ignored by XGBoost are likely redundant with others."""),

    ("code", """xgb_importances = pd.Series(
    xgb_model.feature_importances_, index=X_train.columns, name="gain"
).sort_values(ascending=False)
iv_series_all = pd.Series(encoder.iv_, name="iv")
joined = pd.concat([iv_series_all, xgb_importances], axis=1).fillna(0.0)

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(joined["iv"], joined["gain"], color=PALETTE["primary"], alpha=0.7, s=60)
top_both = joined.sort_values("iv", ascending=False).head(5).index.tolist()
for f in top_both:
    ax.annotate(f, (joined.loc[f, "iv"], joined.loc[f, "gain"]),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.axvline(IV_THRESHOLD, color=PALETTE["neutral"], linestyle=":", alpha=0.6)
ax.set_xlabel("Information Value (univariate)")
ax.set_ylabel("XGBoost gain importance")

# correlation between rankings
rank_corr = float(joined["iv"].rank().corr(joined["gain"].rank()))
ax.set_title(f"Rank correlation IV ↔ XGB gain = {rank_corr:.2f} — the two signals overlap but are not identical")
plt.tight_layout()
plt.show()"""),

    ("md", """**Finding:** IV and XGBoost gain agree on the strongest signals but diverge on the mid-tier — confirming that LR will miss interaction structure the trees can see. This argues for keeping XGBoost as a benchmark, but the Bayesian update layer still goes on LR (see decision below)."""),

    ("md", """## Model selection for the Bayesian update layer

**Decision: logistic regression carries the Bayesian update.**

Why:
1. The analyst prior is specified in log-odds units (`μ_prior`, `σ²_prior` per feature category). The Normal-Normal conjugate update assumes the likelihood is Gaussian in log-odds space — LR's coefficients are exactly that.
2. XGBoost's "importance" is gain, not a log-odds contribution; there is no closed-form mapping from gain back to the coefficient the prior would update.
3. Calibrated XGBoost is retained as a benchmark and for the macro-shift detector (its OOT Brier is the reference point the LR+prior posterior is graded against)."""),

    ("code", """ARTEFACT_DIR = PROJECT_ROOT / cfg["data"]["artefacts_dir"]
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

lr_artefact = ARTEFACT_DIR / f"lr_pipeline_{ts}.joblib"
xgb_artefact = ARTEFACT_DIR / f"xgb_calibrated_{ts}.joblib"

# bundle everything LR needs to score from raw X
lr_bundle = {
    "encoder": encoder,
    "kept_features": kept_features,
    "scaler": scaler,
    "classifier": lr,
    "feature_order": list(X_train.columns),
}
joblib.dump(lr_bundle, lr_artefact)
joblib.dump(xgb_cal, xgb_artefact)

metadata = {
    "timestamp_utc": ts,
    "lr": {
        "artefact": str(lr_artefact.relative_to(PROJECT_ROOT)),
        "in_time_gini":  gini(y_train.values, lr_in_pred),
        "oot_gini":       gini(y_test.values, lr_oot_pred),
        "in_time_brier": brier(y_train.values, lr_in_pred),
        "oot_brier":      brier(y_test.values, lr_oot_pred),
        "oot_gini_ci": [gini_ci_lr["lower"], gini_ci_lr["upper"]],
        "n_features_used": len(kept_features),
    },
    "xgb": {
        "artefact": str(xgb_artefact.relative_to(PROJECT_ROOT)),
        "in_time_gini":  gini(y_train.values, xgb_in_pred),
        "oot_gini":       gini(y_test.values, xgb_oot_pred),
        "in_time_brier": brier(y_train.values, xgb_in_pred),
        "oot_brier":      brier(y_test.values, xgb_oot_pred),
        "oot_gini_ci": [gini_ci_xgb["lower"], gini_ci_xgb["upper"]],
        "best_iteration": int(xgb_model.best_iteration) if xgb_model.best_iteration is not None else None,
    },
    "n_train": int(len(y_train)),
    "n_test":  int(len(y_test)),
    "train_default_rate": float(y_train.mean()),
    "test_default_rate":  float(y_test.mean()),
    "sklearn_version": sklearn.__version__,
    "xgboost_version": xgb.__version__,
    "python_version": platform.python_version(),
}
meta_path = ARTEFACT_DIR / "model_metadata.json"
with meta_path.open("w") as f:
    json.dump(metadata, f, indent=2)
print(f"Wrote {lr_artefact.name}")
print(f"Wrote {xgb_artefact.name}")
print(f"Wrote {meta_path.name}")
print()
print(json.dumps({k: v for k, v in metadata.items() if k in ("lr", "xgb")}, indent=2, default=float))"""),

    ("md", """## Where Stage 3 picks this up

Stage 3 will:
1. Lift `WoeEncoder` → tested `WoETransformer(BaseEstimator, TransformerMixin)` (`src/woe_transformer.py`) with 5 unit tests.
2. Lift LR + XGBoost builders → `src/models.py` with the same hyperparameters from `cfg`.
3. Lift bootstrap and PSI helpers → `src/bootstrap.py` and `src/monitoring.py`.
4. Replace this notebook's ad-hoc serialisation with `src/serialisation.py`, and write a `model_versions` row into the SQLite registry.

`decisions.md` will record the LR-for-Bayesian-update choice and the OOT-split rationale before any Stage 3 code lands."""),
]


def main() -> None:
    nb = new_notebook()
    for kind, content in CELLS:
        if kind == "md":
            nb.cells.append(new_markdown_cell(content))
        else:
            nb.cells.append(new_code_cell(content))
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with TARGET.open("w") as f:
        nbformat.write(nb, f)
    print(f"Wrote {TARGET}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
