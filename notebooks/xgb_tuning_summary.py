"""Marimo notebook: summary of the XGBoost tuning fiddle (Stage 2 follow-up).

Run interactively:
    marimo edit notebooks/xgb_tuning_summary.py

Export to HTML:
    marimo export html notebooks/xgb_tuning_summary.py -o notebooks/exports/xgb_tuning_summary.html
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RESULTS = PROJECT_ROOT / "results"
    ARTEFACTS = PROJECT_ROOT / "artefacts"
    return ARTEFACTS, PROJECT_ROOT, RESULTS, json, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
        # XGBoost tuning — summary

        After the first run of `notebooks/01_data_foundation.ipynb`, XGBoost reported
        `best_iteration = 499` against `n_estimators = 500` — early stopping never
        fired, which meant training was being truncated mid-descent rather than
        converging to a natural plateau. This notebook records the diagnostic sweep
        that fixed it.

        **Setup unchanged from notebook 01:** Polish Bankruptcy, OOT split (train
        years 1–3, test years 4–5), 15% calibration holdout, 15% early-stopping
        holdout inside the modelfit set, same random seed.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 1 — Long run to find the true plateau")
    return


@app.cell
def _(mo):
    mo.md(
        """
        Re-fit with `n_estimators=3000`, `early_stopping_rounds=200` (effectively
        uncapped). The real plateau on the early-stopping holdout sits at iteration
        **1305** — ~2.6× the previous cap. Train log-loss continues to fall past
        that point (overfit signature), but eval log-loss is flat.
        """
    )
    return


@app.cell
def _(RESULTS, mo):
    learning_curve_path = RESULTS / "xgb_learning_curve.png"
    mo.image(str(learning_curve_path), alt="XGBoost learning curve", width=900)
    return (learning_curve_path,)


@app.cell
def _(mo):
    mo.md("## 2 — Grid sweep across `(n_estimators, early_stopping_rounds)`")
    return


@app.cell
def _(RESULTS, pd):
    grid_df = pd.read_csv(RESULTS / "xgb_tune_grid.csv")
    grid_df
    return (grid_df,)


@app.cell
def _(mo):
    mo.md(
        """
        **Reading the grid:**

        - `(500, 30)` — the previous setting. `best_iteration = 499` means the cap
          was the binding constraint, not the early-stopping criterion.
        - `(1000, 30)` — same story: capped at `best=999`.
        - `(1500, 50)` and beyond — `best_iteration` stabilises at 1305, and
          additional headroom does not change OOT metrics. The plateau is real.
        - `early_stopping_rounds` (10 vs 30 vs 50 vs 100) does not matter while the
          cap is binding, and 50 is enough to detect the true plateau cleanly
          without a false trip on a transient eval dip.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 3 — Calibrated XGBoost on the OOT set, before vs after")
    return


@app.cell
def _(pd):
    before_after = pd.DataFrame(
        [
            {
                "metric": "best_iteration",
                "before (n_est=500, es=30)": "499 (capped)",
                "after (n_est=2000, es=50)": "1305 (plateau)",
                "delta": "natural convergence",
            },
            {
                "metric": "OOT Gini",
                "before (n_est=500, es=30)": "0.9064",
                "after (n_est=2000, es=50)": "0.9102",
                "delta": "+0.0038",
            },
            {
                "metric": "OOT Brier",
                "before (n_est=500, es=30)": "0.0272",
                "after (n_est=2000, es=50)": "0.0256",
                "delta": "-0.0016 (-6%)",
            },
            {
                "metric": "OOT Gini 95% bootstrap CI",
                "before (n_est=500, es=30)": "[0.893, 0.919]",
                "after (n_est=2000, es=50)": "[0.896, 0.923]",
                "delta": "shifts up",
            },
            {
                "metric": "ES-holdout log-loss (uncalibrated)",
                "before (n_est=500, es=30)": "0.106",
                "after (n_est=2000, es=50)": "0.071",
                "delta": "-33%",
            },
        ]
    )
    before_after
    return (before_after,)


@app.cell
def _(mo):
    mo.md("## 4 — Verification: current `model_metadata.json`")
    return


@app.cell
def _(ARTEFACTS, json):
    meta = json.loads((ARTEFACTS / "model_metadata.json").read_text())
    xgb_meta = {
        "best_iteration": meta["xgb"]["best_iteration"],
        "oot_gini": round(meta["xgb"]["oot_gini"], 4),
        "oot_brier": round(meta["xgb"]["oot_brier"], 4),
        "oot_gini_ci": [round(meta["xgb"]["oot_gini_ci"][0], 4),
                        round(meta["xgb"]["oot_gini_ci"][1], 4)],
        "artefact": meta["xgb"]["artefact"],
    }
    xgb_meta
    return meta, xgb_meta


@app.cell
def _(mo):
    mo.md(
        """
        ## 5 — Decision

        Committed to `config.yaml`:

        ```yaml
        models:
          xgb:
            n_estimators: 2000          # plateau at ~1305; cap=2000 gives headroom
            early_stopping_rounds: 50   # widened from 30 — eval log-loss only flattens after ~1300 iter
        ```

        Recorded in `decisions.md` (third entry). Confidence: **high** — the plateau
        is unambiguous in the learning curve, the grid sweep is reproducible via
        `scripts/tune_xgb.py`, and the OOT improvement holds through calibration.
        """
    )
    return


if __name__ == "__main__":
    app.run()
