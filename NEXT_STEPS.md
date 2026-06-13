# NEXT_STEPS.md

Work that is still owed on the Expert Prior Engine. For project
context, the file-by-file catalogue of what is already built, and the
standing conventions, see [`CLAUDE.md`](CLAUDE.md). For the history
behind individual design decisions, see [`decisions.md`](decisions.md).

**Current state:** Stages 1–4 complete, 48 tests passing. Stages 5
(Bayesian analyst layer) and 6 (FastAPI + final polish) are unbuilt.

---

## 1 — Decisions still owed

These block or shape the Stage 5 work and should land in `decisions.md`
*before* the corresponding code.

### 1.1 Feature → category mapping (blocks `src/analyst_sim.py`)

The Stage 5 spec assumes four feature categories — `financial_health`,
`payment_behaviour`, `sector_risk`, `management_quality` — but the
Polish dataset is anonymised (`Attr1..Attr64`). We need a defensible
mapping. Two routes:

- **By IV band** — bucket the 64 attrs into 4 groups by IV rank
  (top 16, next 16, etc.) and name them generically (`category_1..4`).
  Cleanest defence: "we don't know what the columns mean, so we
  cluster by signal strength".
- **By domain priors** — look up the original
  [Zięba et al. 2016 paper](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)
  that defines each Attr (it's published) and group by what the ratios
  actually measure. More work but the narrative matches the four
  category names in the spec.

The second route is the one the original CLAUDE.md intended.
Recommendation: spend 30 min mapping by domain, document in
`decisions.md`, then start `src/analyst_sim.py`.

### 1.2 WoE monotonicity (open since Stage 2)

Notebook 00 chart 5 shows 0/5 top features are strictly monotonic
under 10-bin quantile binning. We currently accept this and rely on
LR regularisation. Two paths:

- **Accept (status quo)** — write a `decisions.md` entry justifying
  it and move on.
- **Add monotonic-binning preprocessing** — supervised-tree-based bin
  merging via `optbinning` or equivalent. Refit LR; compare OOT
  Brier; decide if the gain is worth the extra dependency.

Either way, write the entry.

### 1.3 LLN cohort size for nb02 final write-up

Currently `N_PROFILES_DEMO = 6` (15 pairs/model) is tractable in
~10 min but small — the perfect-Gini result for qwen and tinyllama
is partly luck at n=6. For the portfolio version, bump to 10 (45
pairs/model, ~25 min first run; subsequent runs instant due to
SQLite cache). Edit
[`scripts/build_notebook_02.py`](scripts/build_notebook_02.py) and
re-execute.

### 1.4 Larger Ollama models for nb02 / nb03

Currently 4 small local models (qwen 0.5B, tinyllama 1.1B, llama3.2
3.2B, phi3 mini 3.8B). For a portfolio write-up, consider pulling 1–2
mid-size models for family diversity (`mistral:7b`, `gemma2:2b` —
~10 GB disk, ~5x slower per call). Cache amortises the cost on
reruns.

---

## 2 — Stage 5: Bayesian analyst layer

Build in this order; each `src/` module ships with its tests before
the next is written. Spec source: original CLAUDE.md (preserved in
git history at commit 5c44336).

### 2.1 `src/analyst_sim.py`

Single public function `simulate_category_comparisons(X_test, y_test,
feature_categories, cfg, seed)`. For each of the 4 categories, pick
`cfg.analyst_sim.n_companies` OOT companies, generate
`cfg.analyst_sim.n_analysts` synthetic analysts each with a
calibration offset drawn from `N(0, cfg.analyst_sim.heterogeneity_sd)`,
and produce pairwise "which company has worse {category}?" outcomes
probabilistically based on (i) the true category-level risk difference
and (ii) per-analyst noise. Schema matches the `category_comparisons`
SQLite table (already created by `scripts/init_db.py`).

Prerequisite: 1.1 above.

### 2.2 `src/prior_aggregation.py`

Three public functions:

- `apply_forgetting_factor(comparisons, reference_date,
  half_life_days)` — adds a `weight` column `exp(-ln(2)/half_life *
  age_days)`. Half-life from `cfg.analyst_sim.half_life_days` (180).
- `compute_analyst_corrections(comparisons, category)` — fixed-effects
  intercept per analyst (each analyst's mean deviation from the
  global mean).
- `weighted_prior_stats(comparisons, category)` — returns the dict
  consumed by `compute_posterior`: `mu_prior, sigma2_prior,
  precision_prior, n_observations, effective_n`.

### 2.3 `src/posterior_update.py`

- `bootstrap_coefficients(pipeline, X, y, n_bootstrap, seed)` —
  refits the LR pipeline on `n_bootstrap` resamples and returns a
  DataFrame of per-bootstrap coefficient vectors.
- `compute_posterior(prior, mle_estimate, mle_variance, n_defaults,
  cfg)` — Normal-Normal conjugate update. **Thin-data safeguard:**
  if `n_defaults < cfg.analyst_sim.min_defaults_for_mle` (30), set
  `tau_data = 0` so the prior dominates.
- `update_all_coefficients(...)` — loops over categories.

Tests should include: posterior collapses to prior when `tau_data=0`;
posterior collapses to MLE when prior precision → 0; analytic
verification on a known Gaussian conjugate example.

### 2.4 `src/macro_shift.py`

Single function `detect_macro_shift(psi_report, prior_history,
category, psi_threshold, prior_drift_sd_threshold, lookback_days)`.
Fires only when **both** PSI > threshold **and** the analyst-prior
mean has moved > `prior_drift_sd_threshold` SDs from its value
`lookback_days` ago. Single-channel triggers do not fire.

### 2.5 `notebooks/04_bayesian_analyst_layer.ipynb`

Builder script `scripts/build_notebook_04.py` following the pattern
of `build_notebook_02.py`. 4 charts:

1. Prior vs MLE vs posterior coefficients per category, with
   uncertainty bands.
2. Prior weight per category (bar chart) — shows where the prior
   dominates (sparse data) vs where the MLE wins (rich data).
3. Score distribution shift before vs after posterior update.
4. Macro-shift detector timeline on a synthetic shift scenario.

Final cell: end-to-end FastAPI demo — start the server in the
background, make a `/score` call, display the response.

---

## 3 — Stage 6: FastAPI + final polish

### 3.1 `app/main.py`

Load at startup (not per-request):
- the latest LR pipeline (joblib)
- the posterior coefficients (artefact written by 2.3)

Endpoints:
- `POST /score` — accept a borrower profile, override LR coefficients
  with posterior means per category, score, return `predicted_pd,
  model_version, oot_gini, prior_weight_by_category`.
- `GET /health` — `200 OK` with `model_version` and `oot_gini`.
- `GET /metadata` — contents of `artefacts/model_metadata.json`.

### 3.2 `app/schemas.py`

Pydantic input model with one `float` field per Polish dataset column
(64 attrs), plus the `ScoreResponse` output (`predicted_pd: float`,
`model_version: str`, `oot_gini: float`, `prior_weight_by_category:
dict[str, float]`).

### 3.3 `monitoring/retraining_trigger.py`

`evaluate_retraining_need()` runs the PSI + macro-shift check on the
current OOT slice and writes a `retraining_log.jsonl` line whenever a
trigger fires.

### 3.4 `tests/test_api.py`

Four tests via `httpx.AsyncClient`:
- `/health` returns 200.
- `/score` returns a float in [0, 1].
- NaN input → HTTP 422.
- A hand-crafted high-risk profile scores higher than a low-risk one.

---

## 4 — Final checklist (unchecked items only)

From the original spec; checked items have moved to `CLAUDE.md`.

- [ ] `make install && make download && make explore` runs without
      error. *(Consider splitting `explore` into `explore-static` and
      `explore-llm` because nb02 / nb03 are slow on first run.)*
- [ ] `make api` starts FastAPI; `/health` returns 200; `/score`
      returns a PD for a test profile.
- [ ] `decisions.md` entries for: WoE monotonicity (1.2 above);
      feature-category mapping (1.1); forgetting-factor half-life
      rationale (lives in `cfg` but worth a journal entry on why
      180 days); macro-shift dual-trigger rationale.
- [ ] `retraining_log.jsonl` exists with at least one entry —
      produced by §3.3.
- [ ] `make html` re-exports notebooks 00–04 to `results/` (currently
      00–03 only; nb04 doesn't exist yet).

---

## 5 — How to resume

```bash
cd /Users/alexanderfokas/Desktop/repos/bankruptcy-prior-engine
source .venv/bin/activate
make test                  # confirm 48 tests still green
ollama serve &             # if not already running via launchd
```

First action: write the `decisions.md` entry for §1.1
(feature-category mapping), then build `src/analyst_sim.py` against
that mapping. Stage 5 cannot start without it.
