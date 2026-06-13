# NEXT_STEPS.md

Resume-here guide for the Expert Prior Engine. Records the state of the
repo as of the last work session and what to build next.

**Status:** Stages 1, 2, 3 and 4 of [`CLAUDE.md`](CLAUDE.md) are complete.
Stages 5 and 6 are unbuilt.

**Test suite:** 48 tests, all passing (5 WoE + 6 Stage-3 integration + 12
Ollama + 15 Bradley-Terry + 10 POPPER). Run `make test` or
`.venv/bin/python -m pytest tests/`.

**Environment:** project-local `.venv` on Python 3.11.9, installed with
`pip install -e ".[dev]"`. Live Ollama server at `http://localhost:11434`
with four models pulled: `qwen2.5:0.5b`, `tinyllama:latest`,
`llama3.2:latest`, `phi3:mini`.

---

## 1 — What is built (file by file)

### Project root

- **[`CLAUDE.md`](CLAUDE.md)** — Project spec (~42 KB). Single source of
  truth for every design decision. The build is staged with hard
  check-ins; see the "Build Order" section.
- **[`config.yaml`](config.yaml)** — All hyperparameters, paths,
  thresholds, colour palette. The codebase never hardcodes anything that
  lives here.
- **[`pyproject.toml`](pyproject.toml)** — Runtime + dev dependencies
  (pandas, scipy, sklearn 1.8, xgboost 3.2, fastapi, marimo, pytest,
  etc.). Editable install via `pip install -e ".[dev]"`.
- **[`Makefile`](Makefile)** — Targets: `install`, `download`, `initdb`,
  `explore`, `build`, `test`, `api`, `html`, `clean`.
- **[`decisions.md`](decisions.md)** — First-person record of major
  design decisions. Currently three entries: (i) OOT split rationale,
  (ii) LR-as-Bayesian-base, (iii) XGBoost cap raised to
  `n_estimators=2000`, `early_stopping_rounds=50`.
- **[`NEXT_STEPS.md`](NEXT_STEPS.md)** — This file.

### `src/` — production modules

All modules: type hints on every public function, Google-style
docstrings, no `print()` (use `logging.getLogger(__name__)`), every
threshold/hyperparameter read from the `cfg` dict.

- **[`src/__init__.py`](src/__init__.py)** — Empty package marker.
- **[`src/config.py`](src/config.py)** — `load_config(path)`. Validates
  all required top-level keys (`data`, `woe`, `models`, `bootstrap`,
  `psi`, `llm`, `bradley_terry`, `popper`, `analyst_sim`, `api`,
  `colours`) and nested keys.
- **[`src/data_pipeline.py`](src/data_pipeline.py)** —
  `load_and_combine(raw_dir, target_col)` parses all five `Nyear.arff`
  files via `liac-arff`, adds a `year` column, returns
  `(43_405, 66)`. `out_of_time_split(df, ...)` enforces no train/test
  year overlap and logs per-split default rates.
- **[`src/woe_transformer.py`](src/woe_transformer.py)** — Stage 3
  rewrite of the inline `WoeEncoder`. `WoETransformer(BaseEstimator,
  TransformerMixin)` extends bin edges to ±inf at fit, sends NaN to a
  dedicated MISSING bin (index `-1`), warns + falls back to WoE=0 on
  unseen bins, exposes `get_iv_summary()`.
- **[`src/models.py`](src/models.py)** — `build_lr_pipeline(cfg)` →
  WoE → StandardScaler → LR. `build_xgb_pipeline(cfg, n_negative,
  n_positive)` → StandardScaler → XGB with `scale_pos_weight` from
  class counts. `calibrate_xgb(...)` stratifies into modelfit /
  early-stopping / calibration holdouts, fits the pipeline, then fits an
  `IsotonicRegression` on the holdout. `CalibratedXGBPipeline` is the
  resulting sklearn-compatible classifier (sklearn 1.7 removed
  `cv='prefit'` from `CalibratedClassifierCV`; this is the replacement).
  `evaluate_pipeline(...)` returns in-time + OOT Gini/Brier, the
  calibration curve, and the LR coef table.
- **[`src/bootstrap.py`](src/bootstrap.py)** — `bootstrap_ci(y_true,
  y_pred, metric_fn, n_resamples, ci_level, seed)` non-parametric
  percentile CI. `bootstrap_convergence_ci(responses, ...)` resamples
  the *order* of responses for the LLN convergence band.
- **[`src/monitoring.py`](src/monitoring.py)** — `compute_psi(baseline,
  current, edges, epsilon)` uses the fitted `WoETransformer.bin_edges_`
  so out-of-range OOT values land in boundary bins rather than being
  dropped. `monitor_all_features(transformer, X_train, X_current, cfg,
  db_path)` scans every feature, classifies STABLE / MONITOR / RETRAIN,
  and optionally writes to the SQLite `psi_log` table.
- **[`src/serialisation.py`](src/serialisation.py)** — `serialize_model`
  writes the joblib artefact, a per-model metadata JSON, and an
  optional row in `model_versions` (demoting the prior `is_current`
  entry of the same `model_type`). `score_borrowers(df, model_path)` is
  the thin scoring wrapper used by Stage 6's FastAPI endpoint.
- **[`src/ollama_client.py`](src/ollama_client.py)** — Synchronous
  client. `list_available_models(base_url)` lists installed models;
  `query_ollama(model, system_prompt, user_prompt, temperature,
  base_url, timeout, options)` runs a one-shot system + user chat call.
  Three exception classes: `OllamaConnectionError`,
  `OllamaModelNotFoundError`, `OllamaResponseError`.
  `infer_model_family(model_name)` returns a coarse family tag for
  grouping in plots.
- **[`src/bradley_terry.py`](src/bradley_terry.py)** —
  `construct_pairwise_prompts(profiles, task, category)` builds either
  `'borrower_level'` or `'category_level'` pairwise prompts.
  `parse_winner(raw)` regex-extracts A / B from any verbose model
  response. `run_pairwise_experiment(...)` queries every (model, pair)
  combo, persists per-model to SQLite (incrementally so a slow later
  model can't lose the work already done). `fit_bradley_terry(...)`
  fits BT strengths via BFGS MLE (centred so mean = 0).
  `validate_bt_stability(...)` bootstraps to flag high-CV unstable fits.
  `bt_to_log_odds(scores, calibration_curve)` maps strengths through a
  caller-supplied calibration.
- **[`src/popper.py`](src/popper.py)** — `p_to_e_calibrator(p, kappa)`
  Vovk-Wang `e = kappa * p^(kappa-1)`. `sequential_e_accumulation(es,
  alpha)` multiplies the e-values, rejects at first round where
  `E_k >= 1/alpha`. `execute_falsification_experiment(spec, X, y,
  cohort_mask)` runs one of `SUPPORTED_TESTS = {mann_whitney_u,
  fisher_exact, permutation, two_proportion_z}`. Returns `p = 1.0`
  conservatively on any failure (unsupported test, missing column,
  exception). LLM-generated code is **never** `exec()`'d.
  `persist_popper_round(...)` writes to the `popper_experiments` table.

### `notebooks/` — exploration + reporting

- **[`notebooks/_explore_helpers.py`](notebooks/_explore_helpers.py)** —
  Stage 2 inline helpers (WoE/IV, `WoeEncoder`,
  `IsotonicCalibratedClassifier`, PSI, bootstrap). Kept as the
  exploration version — Stage 3 lifted clean equivalents into `src/`.
  Used by `00_data_exploration.ipynb` and `01_data_foundation.ipynb`.
- **[`notebooks/00_data_exploration.ipynb`](notebooks/00_data_exploration.ipynb)**
  → [`results/00_data_exploration.html`](results/00_data_exploration.html)
  (821 KB). 5 charts: class balance by year, missingness heatmap,
  top-IV histograms, IV ranking, WoE monotonicity. Findings: default
  rate climbs 3.9% → 6.9% across years; 3 features have >5% mean
  missingness (top: Attr37 at 43%); 27 features ≥ 0.3 IV; **0/5 top
  features are strictly monotonic** under 10-bin quantile binning
  (flagged for Stage 3 review — see open question below).
- **[`notebooks/01_data_foundation.ipynb`](notebooks/01_data_foundation.ipynb)**
  → [`results/01_data_foundation.html`](results/01_data_foundation.html)
  (740 KB). 5 charts: in-time vs OOT Gini, reliability diagrams,
  bootstrap CI forest plot, PSI heatmap, XGB-gain vs IV scatter.
  Headline OOT numbers: LR Gini 0.7642 / Brier 0.127; XGB Gini 0.9102 /
  Brier 0.0256. CIs do not overlap — XGB statistically dominates on
  discrimination and calibration.
- **[`notebooks/02_llm_lln_experiment.ipynb`](notebooks/02_llm_lln_experiment.ipynb)**
  → [`results/02_llm_lln_experiment.html`](results/02_llm_lln_experiment.html)
  (569 KB). 4 charts: pairwise model agreement matrix, LLN convergence
  band, per-model BT-vs-truth Gini, family bias box plot. Cohort: 6
  OOT borrowers (3 defaulted), top-5-IV features in the prompt. 60
  comparisons cached in SQLite. Striking finding: the two **smallest**
  models (qwen 0.5B, tinyllama 1.1B) perfectly rank the cohort
  (Gini=+1.0); phi3 mini and llama3.2 are anti-aligned (−0.11 and
  −0.33). Implied cohort PDs cluster at 0.50 (true rate); LR
  underestimates this cohort at ~0.30.
- **[`notebooks/03_popper_falsification.ipynb`](notebooks/03_popper_falsification.ipynb)**
  → [`results/03_popper_falsification.html`](results/03_popper_falsification.html)
  (452 KB). 3 charts: p-value trajectory, cumulative e-value vs
  rejection threshold, LLN-vs-POPPER verdict table. All 4 models
  emitted parseable JSON specs. H0 rejected at **round 1**: qwen's
  Fisher test → p ≈ 1e-35 → e = 500,000 → cumulative E = 5e5 > 1/α =
  10. Final cumulative E = 6.25e16 (Stage 2's Gini of 0.91 already
  predicted this — the dataset has obvious learnable signal).
- **[`notebooks/xgb_tuning_summary.py`](notebooks/xgb_tuning_summary.py)**
  → [`notebooks/exports/xgb_tuning_summary.html`](notebooks/exports/xgb_tuning_summary.html)
  (187 KB). Marimo notebook summarising the XGBoost fiddle (see
  `scripts/tune_xgb.py` and the third `decisions.md` entry).

### `scripts/` — one-shot utilities

- **[`scripts/download_data.py`](scripts/download_data.py)** —
  Idempotent fetch from
  `https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip`
  with file size + record count summary. Skips if all five ARFFs are
  already in `data/raw/`.
- **[`scripts/init_db.py`](scripts/init_db.py)** — Creates the six
  SQLite tables with `IF NOT EXISTS`: `borrower_comparisons`,
  `category_comparisons`, `bt_ratings`, `popper_experiments`,
  `psi_log`, `model_versions`. Idempotent.
- **[`scripts/tune_xgb.py`](scripts/tune_xgb.py)** — The XGBoost
  tuning sweep that drove the third `decisions.md` entry. Fits with
  `n_estimators=3000`, `early_stopping_rounds=200` to find the true
  plateau (1305 iterations), plus a grid sweep over (n_estimators,
  early_stopping_rounds). Writes
  [`results/xgb_learning_curve.png`](results/xgb_learning_curve.png)
  and [`results/xgb_tune_grid.csv`](results/xgb_tune_grid.csv).
- **[`scripts/build_notebook_00.py`](scripts/build_notebook_00.py)**,
  **[`scripts/build_notebook_01.py`](scripts/build_notebook_01.py)**,
  **[`scripts/build_notebook_02.py`](scripts/build_notebook_02.py)**,
  **[`scripts/build_notebook_03.py`](scripts/build_notebook_03.py)** —
  Each one constructs an `.ipynb` via `nbformat`. Source-of-truth for
  the notebook content; edit these and rerun, don't edit the
  `.ipynb` JSON by hand.

### `tests/`

- **[`tests/__init__.py`](tests/__init__.py)** — Empty marker.
- **[`tests/conftest.py`](tests/conftest.py)** — Session-scoped `cfg`
  fixture loading `config.yaml`.
- **[`tests/test_woe_transformer.py`](tests/test_woe_transformer.py)**
  — The **5 mandated tests** from CLAUDE.md: no NaN on out-of-range,
  NaN handled, unseen bin warns + WoE=0, IV summary sorted, all-null
  column does not crash.
- **[`tests/test_stage3_integration.py`](tests/test_stage3_integration.py)**
  — 6 real-data smoke tests against the Polish dataset (skip cleanly
  if `data/raw/` is absent). LR pipeline reproduces nb01 OOT Gini;
  calibrated XGB OOT Gini > 0.88 with `best_iteration > 1000` (i.e. the
  tuned cap is the binding constraint); PSI monitor returns sorted
  output; bootstrap CI brackets estimate; convergence CI band narrows;
  serialise → load → score round-trip.
- **[`tests/test_ollama_client.py`](tests/test_ollama_client.py)** — 12
  tests, all HTTP mocked via `monkeypatch`.
- **[`tests/test_bradley_terry.py`](tests/test_bradley_terry.py)** —
  15 tests. Notable: BT recovers known strengths from simulated wins
  (rank correlation = 1).
- **[`tests/test_popper.py`](tests/test_popper.py)** — 10 tests.
  Notable: E[e] ≤ 1.05 across 10k uniform p-values (Vovk-Wang holds
  empirically).

### `data/`, `artefacts/`, `results/`

- `data/raw/{1..5}year.arff` — Polish Bankruptcy ARFFs (already
  downloaded).
- `data/expert_prior_engine.db` — SQLite. Current row counts:
  `borrower_comparisons=60`, `popper_experiments=4`, everything else
  empty (Stage 5 will fill `category_comparisons`, `bt_ratings`,
  `psi_log`, `model_versions`).
- `artefacts/lr_pipeline_<ts>.joblib`,
  `artefacts/xgb_calibrated_<ts>.joblib`,
  `artefacts/model_metadata.json` — Two timestamp snapshots present;
  metadata points to the post-XGB-tuning models from `20260528T183435Z`
  (LR OOT Gini 0.7642, XGB OOT Gini 0.9102).
- `results/` — Five HTML reports + the XGB tuning plot/CSV.

---

## 2 — Key findings and decisions to date

1. **OOT split** (years 1-3 train, 4-5 test) is the right structure
   for this dataset — the year cohort is structurally meaningful.
2. **LR is the base for the Bayesian update**, not XGBoost, because
   the analyst prior is specified in log-odds units and only LR's
   coefficients give a closed-form Normal-Normal posterior. XGBoost
   stays as benchmark + macro-shift input.
3. **XGBoost cap raised to 2000/50.** The plateau is at iteration 1305;
   the previous 500-cap was truncating training mid-descent. Gains:
   OOT Brier −6%, OOT Gini CI shifts up.
4. **WoE non-monotonicity is real** — 0/5 top features are strictly
   monotonic under 10-bin quantile binning. Currently we accept it and
   rely on LR regularisation. See open question 1.
5. **LLM consensus on a 6-borrower cohort beats LR on calibration,**
   but per-model Gini ranges from −0.33 to +1.0 — single-model priors
   are unreliable. This is exactly why Stage 5's prior uses the
   simulated analyst panel, not the LLM panel, for the actual posterior
   update.
6. **POPPER framework works end-to-end** with small local models —
   they all produced valid JSON specs and the cumulative E rejected H0
   at round 1.

---

## 3 — Open questions / decisions still owed

1. **WoE monotonicity** — should the Stage 3 `WoETransformer` add a
   monotonic-binning preprocessing step (supervised-tree-based bin
   merging, e.g. `optbinning`) before LR fitting? Right now we accept
   non-monotonic bins and trust LR regularisation. The Stage 2
   `decisions.md` does not lock this in either direction. Suggested
   experiment: refit LR with monotonic-binned WoE and compare OOT Brier.
2. **LLN cohort size** — `N_PROFILES_DEMO = 6` (15 pairs/model) is
   tractable in ~10 min but small. Bump to 10 (45 pairs/model, ~25 min)
   for the final write-up; perfect-Gini results for two of the four
   models in nb02 are partly luck at n=6.
3. **Model panel for nb02/nb03** — currently 4 small local models.
   For a portfolio write-up, consider pulling 1-2 mid-size models
   (`mistral:7b`, `gemma2:2b`) for more family diversity. Cost: ~10 GB
   disk and slower per-call latency.

---

## 4 — Next steps: Stages 5 and 6

CLAUDE.md prescribes the file layout and signatures. Resume in this
order; each `src/` module should ship with its tests before the next is
built.

### Stage 5 — Bayesian analyst layer

**`src/analyst_sim.py`** — `simulate_category_comparisons(X_test,
y_test, feature_categories, cfg, seed)`. For each of the 4 categories
(`financial_health`, `payment_behaviour`, `sector_risk`,
`management_quality`), pick `cfg.analyst_sim.n_companies` OOT
companies, generate `cfg.analyst_sim.n_analysts` synthetic analysts
each with a calibration offset drawn from
`N(0, cfg.analyst_sim.heterogeneity_sd)`, and produce pairwise
"which company has worse {category}?" outcomes probabilistically based
on (i) the true category-level risk difference and (ii) per-analyst
noise. Schema must match the existing `category_comparisons` SQLite
table. **First step: define the feature → category mapping** — the
spec mentions four categories but the Polish dataset is anonymised
(`Attr1..Attr64`), so this needs a defensible-by-IV-or-domain mapping
written in `decisions.md`.

**`src/prior_aggregation.py`** — Three public functions:

- `apply_forgetting_factor(comparisons, reference_date,
  half_life_days)` adds a `weight` column
  `exp(-ln(2)/half_life * age_days)`.
- `compute_analyst_corrections(comparisons, category)` estimates each
  analyst's mean deviation from the global mean (fixed-effects
  intercept).
- `weighted_prior_stats(comparisons, category)` returns the dict
  consumed by `compute_posterior`: `mu_prior, sigma2_prior,
  precision_prior, n_observations, effective_n`.

**`src/posterior_update.py`** — `bootstrap_coefficients(pipeline,
X, y, n_bootstrap, seed)` returns a DataFrame of per-bootstrap LR
coefficients. `compute_posterior(prior, mle_estimate, mle_variance,
n_defaults, cfg)` does the closed-form Normal-Normal update with the
**thin-data safeguard**: if `n_defaults <
cfg.analyst_sim.min_defaults_for_mle`, set `tau_data = 0` so the prior
dominates. `update_all_coefficients(...)` loops over categories.

**`src/macro_shift.py`** — `detect_macro_shift(psi_report,
prior_history, category, psi_threshold, prior_drift_sd_threshold,
lookback_days)`. Fires only when *both* PSI > threshold *and* the
analyst-prior mean has moved > `prior_drift_sd_threshold` SDs from its
value `lookback_days` ago. Single-channel triggers do not fire.

**`notebooks/04_bayesian_analyst_layer.ipynb`** — 4 charts: prior vs
MLE vs posterior coefficients per category, prior weight bar chart,
score-distribution shift before/after posterior update, macro-shift
detector timeline simulation. Final cell demos the FastAPI endpoint
end-to-end (starts the server, makes a `/score` call).

### Stage 6 — FastAPI + final polish

**`app/main.py`** — FastAPI app loading the latest LR pipeline +
posterior coefficients at startup (not per-request). `POST /score`
overrides LR coefficients with posterior means per category, scores the
borrower, returns `predicted_pd, model_version, oot_gini,
prior_weight_by_category`. `GET /health` → `200` with model version +
OOT Gini. `GET /metadata` returns `model_metadata.json`.

**`app/schemas.py`** — Pydantic input schema with one `float` field
per Polish dataset column, plus the `ScoreResponse` output.

**`monitoring/retraining_trigger.py`** —
`evaluate_retraining_need()` writes a `retraining_log.jsonl` entry
when PSI or macro-shift trips.

**`tests/test_api.py`** — 4 tests via `httpx.AsyncClient`: health
returns 200, score returns float in [0,1], NaN raises 422, high-risk
profile scores higher than low-risk.

### Final checklist (from CLAUDE.md, lifted verbatim)

- [ ] `make install && make download && make explore` runs without
      error (currently the `explore` target presumes nb02 / nb03 are
      slow; consider splitting `explore` into `explore-static` and
      `explore-llm`).
- [x] `make test` passes (48/48 green).
- [x] `make html` exports notebooks 00-03 to `results/`; need nb04.
- [ ] `make api` starts FastAPI; `/health` returns 200; `/score`
      returns a PD for a test profile.
- [x] All charts have finding-first titles.
- [x] All notebooks have markdown narrative before and after every
      chart.
- [ ] `decisions.md` entries for: WoE monotonicity choice (open
      question 1), category mapping (Stage 5 prerequisite), forgetting
      factor parameters (covered by `cfg` but worth a journal entry on
      why 180 days), macro-shift dual-trigger rationale.
- [x] No magic numbers in `src/` or scripts.
- [x] `model_metadata.json` exists in `artefacts/` with non-null OOT
      metrics.
- [ ] `retraining_log.jsonl` exists with at least one entry — produced
      by Stage 6's `monitoring/retraining_trigger.py`.

---

## 5 — How to resume in one command

```bash
cd /Users/alexanderfokas/Desktop/repos/bankruptcy-prior-engine
source .venv/bin/activate
make test                       # confirm 47 tests still green
ollama serve &                  # if not already running via launchd
.venv/bin/python -m pytest tests/  # double-check before starting Stage 5
```

Then read `CLAUDE.md` § "STAGE 5 — Bayesian Analyst Layer + Production"
and start with the feature-category mapping decision (open question 2
above) before writing `src/analyst_sim.py`.
