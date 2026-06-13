# CLAUDE.md — Expert Prior Engine

Project overview, file-by-file catalogue of what's been built, and the
standing conventions. For the work that is still owed, see
[`NEXT_STEPS.md`](NEXT_STEPS.md).

---

## 1 — What this system is

The Expert Prior Engine is a credit-risk scoring system that sits
between human analysts and a statistical model. Three intended parts:

1. **Data foundation** — a logistic regression and a calibrated XGBoost
   on the Polish Bankruptcy dataset, with bootstrapped OOT evaluation
   and PSI monitoring for distribution drift. **Built.**
2. **LLM experiment** — local Ollama models do pairwise borrower
   comparisons (Bradley-Terry) to estimate PD without numeric scales,
   then propose structured falsification experiments under a POPPER-style
   e-value framework. **Built.**
3. **Bayesian analyst layer + production** — simulated analyst panels
   produce a category-level prior, combined with the LR likelihood via
   a closed-form Normal-Normal posterior update, served behind a
   FastAPI endpoint with a PSI/macro-shift retraining trigger. **Not
   built — see `NEXT_STEPS.md`.**

### Audience

- **Primary:** a technical hiring panel reviewing a data-science
  portfolio project. Every design decision is explainable and defensible.
- **Secondary:** product managers evaluating whether to build this as a
  product. The code and notebooks should be clean enough to demo.

### Non-negotiable constraints

These shaped the design and should not be revisited without an explicit
`decisions.md` entry:

- **No numeric LLM scales.** All LLM outputs are pairwise comparisons
  fed to Bradley-Terry. A "1–10" prompt is brittle to LLM weight
  updates; pairwise ordering is stable.
- **Two distinct pairwise tasks.** Part 2 uses *borrower-level*
  comparisons (LLN convergence). Part 3 uses *category-level*
  comparisons (Bayesian prior). Do not conflate.
- **LR is the base for the Bayesian update**, not XGBoost — its
  coefficients are log-odds and have a closed-form Gaussian conjugate
  update. XGBoost stays as benchmark and macro-shift signal. (See
  `decisions.md` entry 2.)
- **Local Ollama only.** No external API calls. Models live in
  `ollama list` on the developer's machine.
- **No magic numbers.** Every threshold, bin count, hyperparameter
  lives in `config.yaml`. The code reads via `src.config.load_config`.
- **LLM code is never `exec()`'d.** In `src/popper.py`, models choose a
  test name from a fixed allow-list (`SUPPORTED_TESTS`) and supply
  inputs; execution happens in trusted code paths.

---

## 2 — Repository structure

```
bankruptcy-prior-engine/
├── CLAUDE.md, NEXT_STEPS.md, decisions.md         # docs
├── README- and reference files
│   ├── config.yaml          # hyperparameters & paths
│   ├── pyproject.toml       # dependencies
│   ├── Makefile             # install / download / test / api / html
│   └── .gitignore           # ignores .venv, artefacts/, data/raw, *.db
├── src/                     # production modules (importable, tested)
├── tests/                   # pytest suite (48 tests, all passing)
├── scripts/                 # one-shot utilities + notebook builders
├── notebooks/               # Jupyter + marimo reports
│   └── exports/             # marimo HTML exports
├── results/                 # executed notebook HTML + plots
├── data/                    # raw ARFFs (gitignored) + sqlite db (gitignored)
└── artefacts/               # joblib models + metadata.json (gitignored)
```

---

## 3 — What is built (file by file)

### Project root

- **[`CLAUDE.md`](CLAUDE.md)** — this file.
- **[`NEXT_STEPS.md`](NEXT_STEPS.md)** — work that is still owed.
- **[`decisions.md`](decisions.md)** — first-person log of major design
  decisions. Three entries: (i) OOT split rationale, (ii) LR as the
  Bayesian-update base, (iii) XGBoost cap raised to `n_estimators=2000`,
  `early_stopping_rounds=50` after the plateau diagnostic.
- **[`config.yaml`](config.yaml)** — single source of truth for every
  hyperparameter, path, threshold, and the colour palette. See § 5
  for the full key reference.
- **[`pyproject.toml`](pyproject.toml)** — runtime + dev dependencies
  (pandas, scipy, sklearn 1.8, xgboost 3.2, fastapi, marimo, pytest,
  jupyter, …). Editable install via `pip install -e ".[dev]"`.
- **[`Makefile`](Makefile)** — targets: `install`, `download`,
  `initdb`, `explore`, `build`, `test`, `api`, `html`, `clean`.

### `src/` — production modules

All modules: type hints on every public function, Google-style
docstrings, no `print()` (use `logging.getLogger(__name__)`), every
threshold read from the `cfg` dict.

- **[`src/__init__.py`](src/__init__.py)** — empty package marker.
- **[`src/config.py`](src/config.py)** — `load_config(path)`. Validates
  required top-level and nested keys.
- **[`src/data_pipeline.py`](src/data_pipeline.py)** —
  `load_and_combine(raw_dir, target_col)` parses all five
  `Nyear.arff` files via `liac-arff`, adds a `year` column, returns a
  `(43_405, 66)` DataFrame. `out_of_time_split(df, ...)` enforces no
  year overlap between train and test and logs per-split default rates.
- **[`src/woe_transformer.py`](src/woe_transformer.py)** —
  `WoETransformer(BaseEstimator, TransformerMixin)`. Bin edges
  extended to ±inf at fit (no NaN on out-of-range), NaN → dedicated
  MISSING bin (`-1`), `UserWarning` + WoE=0 on unseen bins, exposes
  `get_iv_summary()`.
- **[`src/models.py`](src/models.py)** — `build_lr_pipeline(cfg)` →
  WoE → StandardScaler → LR. `build_xgb_pipeline(cfg, n_negative,
  n_positive)` → StandardScaler → XGB with `scale_pos_weight` from
  the class counts. `calibrate_xgb(...)` carves modelfit / early-
  stopping / calibration holdouts, fits, then `IsotonicRegression`
  on the holdout. `CalibratedXGBPipeline` is the sklearn-compatible
  wrapper (replacement for `CalibratedClassifierCV(cv='prefit')`,
  which sklearn 1.7+ removed). `evaluate_pipeline(...)` returns
  in-time + OOT Gini/Brier, calibration curve, and the LR coef table.
- **[`src/bootstrap.py`](src/bootstrap.py)** — `bootstrap_ci(y_true,
  y_pred, metric_fn, n_resamples, ci_level, seed)` percentile CI.
  `bootstrap_convergence_ci(responses, ...)` resamples the *order*
  of responses for the LLN convergence band.
- **[`src/monitoring.py`](src/monitoring.py)** —
  `compute_psi(baseline, current, edges, epsilon)` uses the fitted
  `WoETransformer.bin_edges_` so out-of-range OOT values land in
  boundary bins. `monitor_all_features(transformer, X_train,
  X_current, cfg, db_path)` scans every feature, classifies
  STABLE/MONITOR/RETRAIN, optionally writes to `psi_log`.
- **[`src/serialisation.py`](src/serialisation.py)** — `serialize_model`
  writes a joblib artefact, per-model metadata JSON, and an optional
  row in `model_versions` (demoting the prior `is_current` for the
  same `model_type`). `score_borrowers(df, model_path)` is the thin
  scoring wrapper used by the (not-yet-built) FastAPI endpoint.
- **[`src/ollama_client.py`](src/ollama_client.py)** — synchronous
  client. `list_available_models(base_url)`, `query_ollama(model,
  system_prompt, user_prompt, temperature, base_url, timeout,
  options)`. Three exception classes: `OllamaConnectionError`,
  `OllamaModelNotFoundError`, `OllamaResponseError`.
  `infer_model_family` for plot grouping.
- **[`src/bradley_terry.py`](src/bradley_terry.py)** —
  `construct_pairwise_prompts(profiles, task, category)` builds
  `'borrower_level'` or `'category_level'` prompts. `parse_winner(raw)`
  regex-extracts A/B from verbose responses.
  `run_pairwise_experiment(...)` queries every (model, pair) and
  persists per-model to SQLite (incrementally, so a later slow model
  can't lose earlier work). `fit_bradley_terry(...)` BFGS MLE
  (centred so mean = 0). `validate_bt_stability(...)` bootstrap CV
  check. `bt_to_log_odds(scores, calibration_curve)`.
- **[`src/popper.py`](src/popper.py)** — `p_to_e_calibrator(p, kappa)`
  Vovk-Wang `e = κ · p^(κ-1)`. `sequential_e_accumulation(es, alpha)`
  multiplies e-values, rejects at first `E_k >= 1/α`.
  `execute_falsification_experiment(spec, X, y, cohort_mask)` runs
  one of `SUPPORTED_TESTS = {mann_whitney_u, fisher_exact,
  permutation, two_proportion_z}`. Returns `p = 1.0` conservatively
  on any failure (unsupported test, missing column, exception). LLM
  output never `exec()`'d. `persist_popper_round(...)` writes to
  `popper_experiments`.

### `notebooks/` — exploration + reporting

- **[`notebooks/_explore_helpers.py`](notebooks/_explore_helpers.py)** —
  Stage-2 inline helpers (WoE/IV, `WoeEncoder`,
  `IsotonicCalibratedClassifier`, PSI, bootstrap). Kept as the
  exploration version; Stage 3 lifted clean equivalents into `src/`.
  Used by `00_data_exploration.ipynb` and `01_data_foundation.ipynb`.
- **[`notebooks/00_data_exploration.ipynb`](notebooks/00_data_exploration.ipynb)**
  → [`results/00_data_exploration.html`](results/00_data_exploration.html)
  (821 KB). 5 finding-first charts. Key findings: default rate climbs
  3.9% → 6.9% across years 1–5; 3 features have >5% mean missingness
  (Attr37 at 43%); 27 features ≥ 0.3 IV; **0/5 top features are
  strictly monotonic** under 10-bin quantile binning (flagged for
  Stage-3 review — see open question in NEXT_STEPS.md).
- **[`notebooks/01_data_foundation.ipynb`](notebooks/01_data_foundation.ipynb)**
  → [`results/01_data_foundation.html`](results/01_data_foundation.html)
  (740 KB). 5 charts. Headline OOT numbers: LR Gini 0.7642 / Brier
  0.127; XGB Gini 0.9102 / Brier 0.0256. CIs do not overlap — XGB
  statistically dominates on discrimination and calibration.
- **[`notebooks/02_llm_lln_experiment.ipynb`](notebooks/02_llm_lln_experiment.ipynb)**
  → [`results/02_llm_lln_experiment.html`](results/02_llm_lln_experiment.html)
  (569 KB). Pairwise borrower-level comparisons, 4 local Ollama
  models against a 6-borrower OOT cohort (60 cached comparisons in
  SQLite). The two **smallest** models (qwen 0.5B, tinyllama 1.1B)
  perfectly rank the cohort (Gini = +1.0); phi3 mini and llama3.2 are
  anti-aligned (-0.11 and -0.33). Implied cohort PDs cluster at 0.50
  (true rate); LR underestimates this cohort at ≈0.30.
- **[`notebooks/03_popper_falsification.ipynb`](notebooks/03_popper_falsification.ipynb)**
  → [`results/03_popper_falsification.html`](results/03_popper_falsification.html)
  (452 KB). All 4 models emitted parseable JSON specs. H0 rejected at
  **round 1**: qwen's Fisher test → p ≈ 1e-35 → e = 500,000 →
  cumulative E = 5e5 > 1/α = 10. Final cumulative E = 6.25e16.
- **[`notebooks/xgb_tuning_summary.py`](notebooks/xgb_tuning_summary.py)**
  → [`notebooks/exports/xgb_tuning_summary.html`](notebooks/exports/xgb_tuning_summary.html)
  (187 KB). Marimo notebook summarising the XGBoost tuning fiddle
  (see `scripts/tune_xgb.py` and `decisions.md` entry 3).

### `scripts/` — one-shot utilities

- **[`scripts/download_data.py`](scripts/download_data.py)** —
  idempotent fetch from UCI; skips if all five ARFFs are in
  `data/raw/`.
- **[`scripts/init_db.py`](scripts/init_db.py)** — creates the six
  SQLite tables with `IF NOT EXISTS`. Idempotent.
- **[`scripts/tune_xgb.py`](scripts/tune_xgb.py)** — the XGBoost
  tuning sweep that drove `decisions.md` entry 3. Writes
  [`results/xgb_learning_curve.png`](results/xgb_learning_curve.png)
  and [`results/xgb_tune_grid.csv`](results/xgb_tune_grid.csv).
- **[`scripts/build_notebook_00.py`](scripts/build_notebook_00.py)**,
  **[`scripts/build_notebook_01.py`](scripts/build_notebook_01.py)**,
  **[`scripts/build_notebook_02.py`](scripts/build_notebook_02.py)**,
  **[`scripts/build_notebook_03.py`](scripts/build_notebook_03.py)** —
  build the corresponding `.ipynb` via `nbformat`. Edit these and
  rerun; don't edit the `.ipynb` JSON by hand.

### `tests/` — pytest suite (48 tests, all passing)

- **[`tests/conftest.py`](tests/conftest.py)** — session-scoped `cfg`
  fixture loading `config.yaml`.
- **[`tests/test_woe_transformer.py`](tests/test_woe_transformer.py)**
  — the **5 mandated tests**: no NaN on out-of-range, NaN handled,
  unseen bin warns + WoE=0, IV summary sorted, all-null column does
  not crash.
- **[`tests/test_stage3_integration.py`](tests/test_stage3_integration.py)**
  — 6 real-data smoke tests (skip cleanly if `data/raw/` absent). LR
  pipeline reproduces nb01 OOT Gini; calibrated XGB OOT Gini > 0.88
  with `best_iteration > 1000`; PSI monitor sorted; bootstrap CI
  brackets estimate; convergence CI band narrows; serialise →
  load → score round-trip.
- **[`tests/test_ollama_client.py`](tests/test_ollama_client.py)** —
  12 tests, all HTTP mocked.
- **[`tests/test_bradley_terry.py`](tests/test_bradley_terry.py)** —
  15 tests; notable: BT recovers known strengths from simulated wins.
- **[`tests/test_popper.py`](tests/test_popper.py)** — 10 tests;
  notable: E[e] ≤ 1.05 across 10k uniform p-values (Vovk-Wang holds).

### Data + artefacts (gitignored — regenerable)

- `data/raw/{1..5}year.arff` — Polish Bankruptcy ARFFs from
  [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip).
  Re-fetch with `python scripts/download_data.py`.
- `data/expert_prior_engine.db` — SQLite. Tables: `borrower_comparisons`
  (60 rows from nb02), `popper_experiments` (4 rows from nb03),
  others empty until Stage 5.
- `artefacts/lr_pipeline_<ts>.joblib`,
  `artefacts/xgb_calibrated_<ts>.joblib`,
  `artefacts/model_metadata.json` — current metadata points to the
  post-tuning snapshot `20260528T183435Z` (LR OOT Gini 0.7642, XGB
  OOT Gini 0.9102).

---

## 4 — Standing conventions

Apply to anything new written in this repo.

### Notebook presentation

- Every chart has a **finding-first title** (conclusion, not
  description). "Default rate is highest in year 1" — not "Default
  rate by year".
- Consistent colour palette from `config.yaml`: blue `#2563eb`
  (primary / non-default), red `#dc2626` (risk / default), green
  `#16a34a` (positive / reference / ground truth), grey `#6b7280`
  (neutral), off-white `#f8fafc` (background).
- All axes labelled with units. Light grey gridlines (`alpha=0.3`).
  Figure size `(12, 5)` for single charts, `(12, 8)` for multi-panel.
  Save at DPI 150.
- One markdown cell before each chart (what we're about to see and
  why) and one after (what we found and what it means).
- Notebooks must export cleanly via `jupyter nbconvert --to html
  --execute`. All file paths relative to project root.

### `src/` modules

- One public function per module; private helpers prefixed `_`.
  Exception: a single class per module is OK (see `WoETransformer`).
- Type hints on every public function (args + return).
- Docstrings: one-line summary, Args, Returns, Raises (if applicable).
- No magic numbers — every threshold passed in via `cfg`.
- All imports at the top of the file.
- No `print()` — use `logging.getLogger(__name__)`. `scripts/` may
  print.

### Testing

- `pytest` with no magic numbers in test files — fixtures load
  `config.yaml` via the `cfg` fixture in `tests/conftest.py`.
- Test names: `test_{what_it_tests}` — no `test_1`, `test_a`.
- All WoE transformer tests must pass before any modelling code is
  built (the 5 mandated tests).

### `decisions.md` template

```markdown
## [Date] — [Decision title]

**Decision:** [What was decided]
**Why:** [The specific reason — config key, test output, finding]
**Alternatives considered:** [What else and why rejected]
**Confidence:** [High / Medium / Low]
```

---

## 5 — Reference

### `config.yaml` keys

The current `config.yaml` has these top-level sections (every key
read by `src.config.load_config` and validated against required-key
lists):

- **`data`** — `raw_dir`, `processed_dir`, `artefacts_dir`,
  `results_dir`, `db_path`, `target_col`, `date_col`, `train_years`
  (`[1, 2, 3]`), `test_years` (`[4, 5]`).
- **`woe`** — `n_bins` (10), `iv_threshold` (0.02),
  `laplace_smoothing` (0.5).
- **`models.lr`** — `C`, `solver`, `max_iter`, `class_weight`.
- **`models.xgb`** — `n_estimators` (2000), `max_depth` (4),
  `learning_rate` (0.05), `subsample`, `colsample_bytree`,
  `early_stopping_rounds` (50), `calibration_method` (isotonic),
  `calibration_cv` (prefit), `calibration_val_frac` (0.15).
- **`bootstrap`** — `n_resamples` (1000), `ci_level` (0.95),
  `seed` (42).
- **`psi`** — `stable_threshold` (0.1), `monitor_threshold` (0.2),
  `epsilon` (1e-6).
- **`llm`** — `ollama_base_url`, `temperature` (0),
  `n_borrower_profiles` (10), `n_category_profiles` (8).
- **`bradley_terry`** — `n_bootstrap_validation` (100),
  `cv_threshold` (0.3).
- **`popper`** — `alpha` (0.10), `kappa` (0.50), `max_rounds` (10).
- **`analyst_sim`** — `n_analysts` (5), `n_companies` (20),
  `heterogeneity_sd` (0.3), `half_life_days` (180),
  `min_defaults_for_mle` (30). *(Stage 5 will start consuming these.)*
- **`api`** — `host`, `port`. *(Stage 6.)*
- **`colours`** — see § 4.

### SQLite schema

`data/expert_prior_engine.db` (created by `scripts/init_db.py`). Six
tables, all with `id INTEGER PRIMARY KEY AUTOINCREMENT`:

- **`borrower_comparisons`** — `model_name, model_family,
  profile_a_id, profile_b_id, winner_id, raw_response,
  query_timestamp`. Populated by nb02 (60 rows currently).
- **`category_comparisons`** — `analyst_id, company_a_id,
  company_b_id, category, winner_id, review_date, weight`. Will be
  populated by `src/analyst_sim.py` in Stage 5.
- **`bt_ratings`** — `source, source_id, category, company_id,
  bt_score, fitted_at`. Stage 5.
- **`popper_experiments`** — `round, model_name, experiment_name,
  null_hypothesis, statistical_test, p_value, e_value,
  cumulative_e, decision, run_timestamp`. Populated by nb03 (4 rows).
- **`psi_log`** — `feature, psi_value, status, computed_at`.
  Written by `src.monitoring.monitor_all_features` when given a
  `db_path`.
- **`model_versions`** — `model_type, artefact_path, oot_gini,
  oot_brier, created_at, is_current`. Written by
  `src.serialisation.serialize_model` when given a `db_path`.

---

## 6 — How to run things

```bash
# one-time
.venv/bin/python -m venv .venv
.venv/bin/pip install -e ".[dev]"
python scripts/download_data.py   # fetch the ARFFs
python scripts/init_db.py         # create the SQLite tables

# routine
make test                         # 48 tests, ~10s
make html                         # re-export all notebooks
ollama serve                      # if not already auto-started
```

For Stage 4 notebook reruns: first run does ~60 LLM queries (~10–15
minutes); subsequent runs are instant because comparisons are cached
in SQLite.

---

## 7 — Key findings to date

1. **OOT split** (years 1-3 train, 4-5 test) is the right structure
   for this dataset — the year cohort is structurally meaningful.
2. **LR is the base for the Bayesian update**, not XGBoost.
3. **XGBoost cap raised to 2000/50** after diagnosing that
   `n_estimators=500` was truncating training mid-descent (plateau is
   at iteration 1305).
4. **WoE non-monotonicity is real** — 0/5 top features are strictly
   monotonic under 10-bin quantile binning. Open question; see
   `NEXT_STEPS.md`.
5. **LLM consensus on a 6-borrower cohort beats LR on calibration**,
   but per-model Gini ranges from -0.33 to +1.0 — single-model priors
   are unreliable. This is exactly why Stage 5's prior will use the
   simulated analyst panel, not the LLM panel.
6. **POPPER framework works end-to-end** with small local models —
   they all produced valid JSON specs and the cumulative E rejected
   H0 at round 1.
