# Decisions

First-person record of major design decisions, in the order they were made.

---

## 2026-05-28 — Out-of-time split (years 1-3 train, 4-5 test)

**Decision:** Split by `year` rather than random 80/20. Years 1, 2, 3 form the training set; years 4 and 5 form the OOT test set.

**Why:** Credit risk is fundamentally a temporal problem. A random split would let the model see year-4 and year-5 examples at training time, hiding the cohort drift that we know exists in this dataset — default rate climbs from 3.9% (year 1) to 6.9% (year 5). The OOT cohort has a higher base rate than the train cohort (5.89% vs 4.21%), which is the conservative direction for evaluation: the model is graded on the harder, higher-imbalance period.

**Alternatives considered:**
- Random stratified 80/20 — rejected because it overstates portability. We saw evidence in notebook 00 that per-year default rate is a structural feature of the dataset, not noise.
- Roll-forward CV (train years 1-2, test year 3; train 1-3, test 4; etc.) — would give a more honest variance estimate but adds complexity not justified at this stage. May revisit during Stage 3 model evaluation.

**Confidence:** High. The temporal structure is explicit in the data and the OOT cohort is realistic.

---

## 2026-05-28 — Logistic regression as the Bayesian update base, XGBoost as benchmark

**Decision:** The Bayesian update layer (Stage 5) will adjust **logistic regression** coefficients with the analyst prior. Calibrated XGBoost is kept as a benchmark and as input to the macro-shift detector, not as the base for the prior update.

**Why:** The analyst prior is specified in log-odds units (`μ_prior`, `σ²_prior` per feature category) and the Normal-Normal conjugate update assumes the likelihood is Gaussian in log-odds space. LR's coefficients are exactly that — there is a closed-form correspondence between the prior, the MLE, and the posterior. XGBoost has no such mapping: its "importance" is a gain statistic, not a log-odds contribution, and there is no defensible way to apply a Gaussian prior to it.

This decision is made *despite* XGBoost dominating LR on raw OOT metrics in notebook 01 (XGB OOT Gini 0.91 vs LR OOT Gini 0.76; XGB OOT Brier 0.027 vs LR Brier 0.127, with bootstrap CIs that do not overlap). The point of the prior update is to close that gap by injecting expert knowledge into the LR coefficients; the XGB Brier becomes the target the posterior is graded against.

**Alternatives considered:**
- Use XGBoost as the base, with a stacked LR-on-XGBoost-leaves layer that the prior updates — adds interpretation indirection and breaks the closed-form posterior.
- Use a Bayesian neural net or full MCMC — overkill for a 64-feature tabular problem, and removes the interpretability that makes the system defensible to a credit committee.

**Confidence:** High on the architectural choice. Medium on whether the LR+prior posterior will actually approach XGB's Brier — that is the empirical question Stage 5 has to answer.

---

## 2026-05-28 — XGBoost cap: `n_estimators=2000`, `early_stopping_rounds=50`

**Decision:** Raised `cfg.models.xgb.n_estimators` from 500 → 2000 and `early_stopping_rounds` from 30 → 50.

**Why:** The Stage 2 fit reported `best_iteration=499` against a 500-iteration cap, meaning early stopping never fired — training was truncated mid-descent. I ran `scripts/tune_xgb.py`, which fits with `n_estimators=3000`, `early_stopping_rounds=200` (essentially uncapped) and traces the train + eval log-loss curves. The natural plateau on the early-stopping holdout sits at **iteration 1305** (eval log-loss flattens; train continues to fall → overfit). Letting the model run to plateau improves:
- OOT Gini (calibrated): 0.9064 → 0.9102
- OOT Brier (calibrated): 0.0272 → 0.0256 (−6%)
- OOT Gini 95% bootstrap CI: [0.893, 0.919] → [0.896, 0.923]
- ES-holdout log-loss (uncalibrated): 0.106 → 0.071 (−33%)

`n_estimators=2000` gives ~700 iterations of headroom past the observed plateau, which is enough margin if a future data refresh shifts the plateau later. `early_stopping_rounds=50` was chosen because the eval loss is monotonically falling for ~1300 iterations — a 30-round patience risks a false stop on a transient eval dip; 50 is roughly the natural "stickiness" of the eval curve near the plateau.

**Alternatives considered:**
- Keep `n_estimators=500` for compute budget — rejected; the 500-iteration model is demonstrably under-trained and would be a misleading benchmark for the LR+prior comparison.
- `n_estimators=1500` — would also work (the plateau is at 1305), but gives only 195 iterations of slack past best_iter, which is too tight for a robust ES trigger across data refreshes.
- Lower learning rate + same cap — would change the plateau location and require a separate sweep; out of scope for this fiddle.

**Artefacts:** Learning curve at `results/xgb_learning_curve.png`; grid sweep at `results/xgb_tune_grid.csv`.

**Confidence:** High. The tuning script is reproducible and the plateau is unambiguous in the learning curve.

---
