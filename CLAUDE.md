# CLAUDE.md — Expert Prior Engine

## Project Context

### What this system does

The Expert Prior Engine is a credit risk system that sits between human analysts and a statistical scoring model. It has three parts built in sequence:

**Part 1 — Data Foundation:** Trains a logistic regression and XGBoost model on the Polish Bankruptcy dataset, produces a serialised sklearn Pipeline, evaluates both models with bootstrapped confidence intervals, and sets up PSI monitoring for distribution drift.

**Part 2 — LLM Experiment:** Queries local Ollama models using pairwise borrower comparisons (Bradley-Terry) to estimate PD without numerical scales — avoiding score drift from model weight updates. Demonstrates the Law of Large Numbers (group mean converging toward ground truth as k increases), then extends this with a POPPER-inspired sequential falsification framework where LLMs propose structured falsification experiments executed against real data, with e-values accumulated for statistically valid rejection decisions.

**Part 3 — Bayesian Analyst Layer + Production:** Simulates credit analyst pairwise comparisons at the feature *category* level (financial health, payment behaviour, sector risk, management quality), fits Bradley-Terry per category, applies an exponential forgetting factor and analyst heterogeneity correction, and combines the resulting prior with the MLE likelihood via a closed-form Normal-Normal posterior update. Deploys the result as a FastAPI scoring endpoint with PSI-triggered retraining detection.

### Who it is for

**Primary audience:** A technical hiring panel reviewing a data science portfolio project. Every design decision should be explainable and defensible. Every chart should tell a clear story. The codebase should read as if written by someone who understands production constraints, not just notebook prototyping.

**Secondary audience:** Product managers evaluating whether to build this as a product. The product story (in the Notion page) addresses them; the code and notebooks should be clean enough to demo.

### What success looks like

1. **Working end-to-end codebase** — all three stages run from `make all` with no manual intervention
2. **Polished notebooks** — each notebook exports to HTML and tells a clear story with finding-first chart titles and markdown narrative
3. **Deployed API** — FastAPI endpoint returns a PD score and accepts a borrower profile as JSON input, running locally
4. **Honest evaluation** — out-of-time Gini and Brier score are the primary metrics, never in-time. Bootstrap CIs are reported alongside point estimates. The POPPER experiment reports both the rejection decision and the Type-I error guarantee.

### Key constraints

- **Score drift:** Never use raw numeric scales (1–N) for LLM outputs. Always use pairwise comparisons with Bradley-Terry. This is non-negotiable — it is the architectural decision that makes the system robust to LLM weight updates.
- **Two separate pairwise tasks:** Part 2 uses holistic borrower-level comparisons ("which company is more likely to default?") for the LLN convergence experiment. Part 3 uses category-level comparisons ("which company has worse financial health?") for the Bayesian prior. Do not conflate these.
- **XGBoost is the tree model:** Do not substitute random forest or other ensembles without being asked.
- **Local Ollama only:** No external API calls. Models available via `ollama list` on the local machine.
- **Staged build with check-ins:** Build Stages 1 and 2 (environment + exploration notebooks), then stop and wait for review before proceeding to Stage 3 (src/ modules). Never build ahead without explicit approval.
- **No magic numbers:** Everything reads from `config.yaml`. The code should never contain a hardcoded threshold, bin count, or learning rate.
- **Hard stop after exploration notebooks:** Do not build the scoring engine (any of Stage 3 onwards) until the data has been explored, the WoE maps have been reviewed for monotonicity, and the IV rankings have been validated.

---

## Repository Structure

```
expert_prior_engine/
├── CLAUDE.md                    # this file
├── Makefile                     # make install | make explore | make build | make test | make api
├── config.yaml                  # all hyperparameters, paths, thresholds
├── pyproject.toml               # dependencies
├── decisions.md                 # first-person design rationale (write as you build)
│
├── data/
│   ├── raw/                     # 1year.arff ... 5year.arff (download script creates these)
│   └── processed/               # combined_df.parquet (created by load script)
│
├── artefacts/                   # serialised models and metadata (gitignored)
│
├── results/                     # plots, HTML notebook exports, findings reports
│
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_data_foundation.ipynb
│   ├── 02_llm_lln_experiment.ipynb
│   ├── 03_popper_falsification.ipynb
│   └── 04_bayesian_analyst_layer.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                # loads config.yaml, validates all keys
│   ├── data_pipeline.py         # load_and_combine(), out_of_time_split()
│   ├── woe_transformer.py       # WoETransformer (sklearn-compatible)
│   ├── models.py                # build_lr_pipeline(), build_xgb_pipeline(), evaluate_pipeline()
│   ├── bootstrap.py             # bootstrap_ci(), bootstrap_convergence_ci()
│   ├── monitoring.py            # compute_psi(), monitor_all_features()
│   ├── serialisation.py         # serialize_model(), score_borrowers()
│   ├── ollama_client.py         # query_ollama(), list_available_models()
│   ├── bradley_terry.py         # run_pairwise_experiment(), fit_bradley_terry(), bt_to_log_odds()
│   ├── popper.py                # p_to_e_calibrator(), sequential_e_accumulation(), execute_falsification()
│   ├── analyst_sim.py           # simulate_analyst_reviews(), simulate_category_comparisons()
│   ├── prior_aggregation.py     # apply_forgetting_factor(), compute_analyst_corrections(), weighted_prior_stats()
│   ├── posterior_update.py      # compute_posterior(), update_all_coefficients(), bootstrap_coefficients()
│   └── macro_shift.py           # detect_macro_shift() — fires when PSI > threshold AND prior drift > 1 SD
│
├── app/
│   ├── main.py                  # FastAPI scoring endpoint
│   └── schemas.py               # Pydantic input/output models
│
├── monitoring/
│   └── retraining_trigger.py    # evaluate_retraining_need(), logs to retraining_log.jsonl
│
├── tests/
│   ├── test_woe_transformer.py  # 5 unit tests (out-of-range, missing, unseen, iv sort, null col)
│   ├── test_bradley_terry.py
│   ├── test_popper.py
│   ├── test_posterior_update.py
│   └── test_api.py
│
└── scripts/
    └── download_data.py         # downloads Polish Bankruptcy dataset from UCI
```

---

## Configuration

All values live in `config.yaml`. The code must never hardcode these values — always import via `src/config.py`.

```yaml
# config.yaml

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  artefacts_dir: "artefacts"
  results_dir: "results"
  target_col: "class"
  date_col: "year"
  train_years: [1, 2, 3]
  test_years: [4, 5]

woe:
  n_bins: 10
  iv_threshold: 0.02       # features below this IV are dropped
  laplace_smoothing: 0.5

models:
  lr:
    C: 1.0
    solver: "lbfgs"
    max_iter: 1000
    class_weight: "balanced"
  xgb:
    n_estimators: 500
    max_depth: 4
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 30
    calibration_method: "isotonic"
    calibration_cv: "prefit"
    calibration_val_frac: 0.15

bootstrap:
  n_resamples: 1000
  ci_level: 0.95
  seed: 42

psi:
  stable_threshold: 0.1
  monitor_threshold: 0.2    # above this: RETRAIN flag
  epsilon: 1e-6

llm:
  ollama_base_url: "http://localhost:11434"
  temperature: 0
  n_borrower_profiles: 10   # number of profiles for pairwise matrix
  n_category_profiles: 8    # number of profiles per feature category (Part 3)

bradley_terry:
  n_bootstrap_validation: 100   # validate BT stability
  cv_threshold: 0.3             # flag insufficient comparisons if CV > this

popper:
  alpha: 0.10
  kappa: 0.50               # Vovk-Wang calibrator parameter
  max_rounds: 10

analyst_sim:
  n_analysts: 5
  n_companies: 20
  heterogeneity_sd: 0.3     # SD of analyst-level noise (in log-odds units)
  half_life_days: 180
  min_defaults_for_mle: 30  # below this, set tau_data = 0 (prior dominates)

api:
  host: "0.0.0.0"
  port: 8000

colours:
  primary: "#2563eb"
  secondary: "#dc2626"
  tertiary: "#16a34a"
  neutral: "#6b7280"
  background: "#f8fafc"
```

---

## Database Schema

Use SQLite (`data/expert_prior_engine.db`) for storing analyst reviews and experiment results.

```sql
-- Pairwise comparisons from LLM borrower-level experiment (Part 2)
CREATE TABLE borrower_comparisons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    model_family    TEXT NOT NULL,
    profile_a_id    TEXT NOT NULL,
    profile_b_id    TEXT NOT NULL,
    winner_id       TEXT NOT NULL,    -- 'A' or 'B'
    raw_response    TEXT,
    query_timestamp TEXT NOT NULL
);

-- Category-level analyst pairwise comparisons (Part 3)
CREATE TABLE category_comparisons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analyst_id      TEXT NOT NULL,
    company_a_id    TEXT NOT NULL,
    company_b_id    TEXT NOT NULL,
    category        TEXT NOT NULL,    -- 'financial_health' | 'payment_behaviour' | 'sector_risk' | 'management_quality'
    winner_id       TEXT NOT NULL,    -- 'A' or 'B' (higher risk winner)
    review_date     TEXT NOT NULL,
    weight          REAL DEFAULT 1.0  -- populated by forgetting factor
);

-- Bradley-Terry fitted ratings (one row per model/analyst × category)
CREATE TABLE bt_ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,    -- 'llm' | 'analyst_sim'
    source_id       TEXT NOT NULL,    -- model_name or analyst_id
    category        TEXT NOT NULL,
    company_id      TEXT NOT NULL,
    bt_score        REAL NOT NULL,
    fitted_at       TEXT NOT NULL
);

-- POPPER falsification experiment results (Part 2)
CREATE TABLE popper_experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round           INTEGER NOT NULL,
    model_name      TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    null_hypothesis TEXT NOT NULL,
    statistical_test TEXT NOT NULL,
    p_value         REAL NOT NULL,
    e_value         REAL NOT NULL,
    cumulative_e    REAL NOT NULL,
    decision        TEXT NOT NULL,    -- 'CONTINUE' | 'REJECT'
    run_timestamp   TEXT NOT NULL
);

-- PSI monitoring log
CREATE TABLE psi_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feature         TEXT NOT NULL,
    psi_value       REAL NOT NULL,
    status          TEXT NOT NULL,    -- 'STABLE' | 'MONITOR' | 'RETRAIN'
    computed_at     TEXT NOT NULL
);

-- Model version registry
CREATE TABLE model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type      TEXT NOT NULL,    -- 'lr' | 'xgb' | 'posterior'
    artefact_path   TEXT NOT NULL,
    oot_gini        REAL,
    oot_brier       REAL,
    created_at      TEXT NOT NULL,
    is_current      INTEGER DEFAULT 0
);
```

---

## Build Order

Build in stages. After each stage, stop and wait for review before proceeding.

---

### STAGE 1 — Environment + Data (Build First)

**Goal:** Everything installs cleanly and the data loads correctly.

#### Tasks

**1.1 `pyproject.toml`**
Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `liac-arff`, `fastapi`, `uvicorn`, `pydantic`, `joblib`, `matplotlib`, `seaborn`, `httpx`, `pytest`, `pyarrow`.

**1.2 `Makefile`**
Targets:
- `make install` — `pip install -e .`
- `make download` — runs `scripts/download_data.py`
- `make explore` — runs notebooks 00 and 01 via `jupyter nbconvert --execute`
- `make build` — runs all src/ modules and tests (Stage 3+)
- `make test` — `pytest tests/`
- `make api` — `uvicorn app.main:app --reload`
- `make html` — exports all executed notebooks to `results/`

**1.3 `scripts/download_data.py`**
Downloads the Polish Bankruptcy dataset from UCI:
```
https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip
```
Extracts to `data/raw/`. Prints file sizes and record counts.

**1.4 `src/config.py`**
```python
def load_config(path: str = "config.yaml") -> dict:
    """Load and validate config.yaml. Raises ValueError for missing required keys."""
```
Validate all required top-level keys exist. Return the config dict. Import as `from src.config import load_config; cfg = load_config()`.

**1.5 `src/data_pipeline.py`**
```python
def load_and_combine(raw_dir: str, target_col: str) -> pd.DataFrame:
    """
    Load all ARFF files from raw_dir, parse with liac-arff, add 'year' column (1-5),
    combine into a single DataFrame. Target column renamed to target_col.
    
    Returns: DataFrame with shape ~(10000, 66) including 'year' and target_col.
    """

def out_of_time_split(
    df: pd.DataFrame,
    date_col: str,
    train_years: list[int],
    test_years: list[int],
    target_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split by year column into train and test sets.
    
    Returns: X_train, y_train, X_test, y_test
    Raises: AssertionError if date ranges overlap.
    Prints: n, default_rate, date_range for each split.
    """
```

**1.6 Database initialisation**
```python
# scripts/init_db.py
def init_database(db_path: str) -> None:
    """Create all tables in the SQLite database. Idempotent (uses CREATE TABLE IF NOT EXISTS)."""
```

**CHECK IN after Stage 1.** Confirm data loaded correctly before proceeding.

---

### STAGE 2 — Exploration Notebooks

**Goal:** Understand the data before building any model. Produce polished notebooks that export to HTML.

#### ⚠️ HARD STOP AFTER STAGE 2 ⚠️
Do not build any src/ modules or models until Stage 2 notebooks have been reviewed and approved.

---

#### Notebook 00: Data Exploration

**File:** `notebooks/00_data_exploration.ipynb`
**Question it answers:** What is this dataset, what does the class imbalance look like, which features have strong IV, and are there any data quality issues that need addressing before modelling?

**Structure:**
1. Markdown: dataset provenance, years 1-5 description, what `class=1` means
2. Load data, print shape, dtypes, memory usage
3. **Chart 1 — Class balance by year:** Stacked bar chart, one bar per year, showing default rate. Finding-first title: *"Default rate is highest in year 1 (X%) and drops in later years — a selection effect"*
4. **Chart 2 — Missing value heatmap:** Seaborn heatmap of missingness by feature and year. Title: *"X features have >Y% missing values — these will be assigned a MISSING WoE bin"*
5. **Chart 3 — Feature distribution examples:** 4-panel plot of the 4 highest-IV features (after WoE fitting), showing histogram by class. Title: *"ROA separates bankrupt and non-bankrupt companies most cleanly (IV=X)"*
6. **Chart 4 — IV ranking bar chart:** Top 20 features by IV, coloured by IV band (useless/weak/medium/strong). Title: *"X features have strong IV (>0.3); Y are useless (<0.02) and will be dropped"*
7. **Chart 5 — WoE monotonicity check:** For top 5 features, plot WoE value by bin. Flag non-monotonic features. Title: *"WoE is monotonic for X/5 top features — non-monotonic features need bin review"*
8. Markdown summary cell: key findings, features to investigate, any data quality flags

**Polished version looks like:**
- Consistent colour palette (primary=#2563eb for non-default, secondary=#dc2626 for default)
- All axes labelled with units
- Annotations on charts where appropriate (e.g. IV threshold lines)
- Clean markdown narrative before and after every chart
- Final cell: `pd.DataFrame({'finding': [...], 'action': [...]})` printed as a table

**Export:** `make html` runs `jupyter nbconvert --to html --execute notebooks/00_data_exploration.ipynb --output results/00_data_exploration.html`

---

#### Notebook 01: Data Foundation

**File:** `notebooks/01_data_foundation.ipynb`
**Question it answers:** How do logistic regression and XGBoost compare on out-of-time performance, and which is the right baseline for the Bayesian update layer?

**Structure:**
1. Markdown: recap of OOT split rationale, why not random 80/20
2. Fit both models using `src/models.py` functions
3. **Chart 1 — OOT vs in-time Gini comparison:** Grouped bar chart. Title: *"Both models show Gini drop from in-time to OOT — LR drops X points, XGBoost drops Y points"*
4. **Chart 2 — Reliability diagrams:** Side-by-side for LR and XGBoost (calibrated). Title: *"LR is better calibrated than XGBoost at extreme probabilities"* (or vice versa — let the data decide)
5. **Chart 3 — Bootstrap CI comparison:** Forest plot. LR vs XGBoost, Gini and Brier side by side with 95% CIs. Title: *"Confidence intervals overlap on Gini — performance difference is not statistically significant"* (or the opposite)
6. **Chart 4 — PSI heatmap:** Feature PSI values (OOT vs train), colour-coded by threshold. Title: *"X features show significant distribution shift between training and OOT period"*
7. **Chart 5 — XGBoost feature importance vs IV ranking:** Scatter plot of gain-based importance vs IV. Title: *"Features ranked highly by both IV and XGBoost gain are the most robust predictors"*
8. Markdown: model selection decision and rationale. Which model becomes the basis for the Bayesian layer and why. (LR is preferred for the Bayesian update because its coefficients have direct log-odds interpretation, matching the prior structure.)
9. Serialise both models: `src/serialisation.py`
10. Print model_metadata.json

**Polished version looks like:**
- Every chart has a finding-first title (conclusion, not description)
- Reliability diagrams have a diagonal reference line
- Bootstrap CI chart annotates where CIs overlap
- Final cell produces `decisions.md` entry for the model choice

**Export:** `results/01_data_foundation.html`

---

### ⚠️ CHECK IN HERE — Do not proceed to Stage 3 until Stage 2 is reviewed ⚠️

---

### STAGE 3 — src/ Modules: Data Foundation

Build the src/ modules that back the Stage 2 notebooks. These should be clean, tested, importable code — not notebook cells copy-pasted.

#### `src/woe_transformer.py`

```python
class WoETransformer(BaseEstimator, TransformerMixin):
    """
    WoE transformer compatible with sklearn Pipeline.
    
    Design decisions:
    - Bin edges extended to ±inf at inference to prevent pd.cut NaN on out-of-range values
    - Missing values → 'MISSING' bin (never silently zeroed)
    - Unseen categories → WoE=0.0 with UserWarning
    - Laplace smoothing (+smoothing param) prevents log(0) in sparse bins
    - IV computed per feature; accessible via get_iv_summary()
    """
    
    def __init__(self, n_bins: int = 10, laplace_smoothing: float = 0.5):
        ...
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer": ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def get_iv_summary(self) -> pd.DataFrame: ...
```

Tests (all 5 must pass before any other module is built):
1. `test_no_nan_on_out_of_range` — values at 1e9 must not produce NaN
2. `test_missing_values_handled` — NaN inputs produce valid WoE output
3. `test_unseen_category_warns_and_returns_zero` — UserWarning + WoE=0.0
4. `test_iv_summary_sorted_descending` — IV table must be sorted correctly
5. `test_all_null_column_does_not_crash` — transformer must not raise

#### `src/models.py`

```python
def build_lr_pipeline(cfg: dict) -> Pipeline:
    """
    Build LogisticRegression Pipeline: WoETransformer → StandardScaler → LogisticRegression.
    All hyperparameters read from cfg['models']['lr'].
    """

def build_xgb_pipeline(cfg: dict, n_negative: int, n_positive: int) -> Pipeline:
    """
    Build XGBoost Pipeline: StandardScaler → XGBClassifier.
    scale_pos_weight computed from n_negative / n_positive.
    Early stopping requires eval_set; handled internally.
    Not wrapped with CalibratedClassifierCV here — calibration done separately in calibrate_xgb().
    """

def calibrate_xgb(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict
) -> CalibratedClassifierCV:
    """
    Hold out cfg['models']['xgb']['calibration_val_frac'] of training data.
    Fit base_pipeline on the remainder.
    Apply CalibratedClassifierCV(cv='prefit', method=cfg['models']['xgb']['calibration_method']).
    Returns fitted calibrated pipeline.
    """

def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Fit pipeline on train, evaluate on both train (in-time) and test (OOT).
    Returns dict with: in_time_gini, oot_gini, in_time_brier, oot_brier,
    calibration_curve_data, coef_table (LR only).
    """
```

#### `src/bootstrap.py`

```python
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int,
    ci_level: float,
    seed: int
) -> dict:
    """Non-parametric bootstrap CI for any scalar metric. Returns estimate, lower, upper, std."""

def bootstrap_convergence_ci(
    responses: list[float],
    n_resamples: int,
    ci_level: float,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap CI bands on an LLN convergence curve.
    Resamples the order of responses, recomputes running mean trajectory.
    Returns lower_bound, upper_bound arrays of length n.
    """
```

#### `src/monitoring.py`

```python
def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    edges: np.ndarray,
    epsilon: float
) -> float:
    """PSI for one feature. Edges from WoETransformer.bin_edges_. Extends to ±inf internally."""

def monitor_all_features(
    transformer: WoETransformer,
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    cfg: dict
) -> pd.DataFrame:
    """
    Compute PSI for every numeric feature. Returns DataFrame: feature, psi, status.
    Writes results to SQLite psi_log table.
    """
```

#### `src/serialisation.py`

```python
def serialize_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    cfg: dict
) -> Path:
    """
    Persist fitted Pipeline with joblib (timestamped filename).
    Write model_metadata.json: timestamp, model_type, features, n_test, oot_gini, oot_brier,
    sklearn_version, python_version.
    Returns path to saved model.
    """

def score_borrowers(df: pd.DataFrame, model_path: str) -> pd.Series:
    """Load serialised Pipeline. Score df. Return pd.Series of predicted PDs."""
```

**CHECK IN after Stage 3.** Run `make test` — all tests must pass.

---

### STAGE 4 — LLM Experiment Notebooks + src/

**Goal:** Run the LLN convergence and POPPER experiments. Produce the Part 2 notebook.

#### `src/ollama_client.py`

```python
def list_available_models(base_url: str) -> list[str]:
    """Return list of model names available from local Ollama instance."""

def query_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    base_url: str
) -> str:
    """
    Query a local Ollama model. Returns raw string response.
    Raises OllamaConnectionError if server is unreachable.
    Raises OllamaModelNotFoundError if model is not available.
    """
```

#### `src/bradley_terry.py`

```python
def construct_pairwise_prompts(
    profiles: list[dict],
    task: str,      # 'borrower_level' | 'category_level'
    category: str   # only used when task='category_level'
) -> list[tuple[str, str, str]]:
    """
    For each pair (i,j) of profiles, return (profile_a_json, profile_b_json, prompt).
    For borrower_level: 'Which of these two companies is more likely to default?'
    For category_level: 'Which of these two companies has worse {category}?'
    Returns list of (profile_a_id, profile_b_id, prompt) tuples.
    """

def run_pairwise_experiment(
    models: list[str],
    profiles: list[dict],
    task: str,
    category: str,
    cfg: dict,
    db_path: str
) -> pd.DataFrame:
    """
    Query each model with each pairwise prompt. Parse winner (A or B).
    Write results to SQLite. Returns DataFrame of all comparison results.
    """

def fit_bradley_terry(comparisons: pd.DataFrame, item_col_a: str, item_col_b: str, winner_col: str) -> pd.Series:
    """
    Fit Bradley-Terry MLE on pairwise comparison data.
    Uses scipy.optimize.minimize on the negative log-likelihood.
    Returns pd.Series: item_id → BT score (unnormalised log-strength).
    """

def validate_bt_stability(
    comparisons: pd.DataFrame,
    n_bootstrap: int,
    cv_threshold: float
) -> dict:
    """
    Bootstrap BT parameters (resample comparisons, refit n_bootstrap times).
    Return: is_stable (bool), mean_cv (float), per-item CV.
    Warn if is_stable is False.
    """

def bt_to_log_odds(
    bt_scores: pd.Series,
    calibration_curve: Callable
) -> pd.Series:
    """
    Map Bradley-Terry scores to log-odds via calibration curve.
    calibration_curve: fitted scipy function mapping BT percentile → log-odds.
    Calibration is fitted against empirical default rates by BT score band,
    NOT against LR predicted probabilities (avoids circular dependency).
    """
```

#### `src/popper.py`

```python
def p_to_e_calibrator(p_value: float, kappa: float) -> float:
    """Vovk-Wang calibrator. e = kappa * p^(kappa-1). E[e] <= 1 under H0 unconditionally."""

def sequential_e_accumulation(e_values: list[float], alpha: float) -> dict:
    """
    Multiply e-values sequentially. Reject when product >= 1/alpha.
    Returns: cumulative_E list, final_E, threshold, rejected (bool), rejected_at_round.
    """

SUPPORTED_TESTS = {"mann_whitney_u", "fisher_exact", "permutation", "two_proportion_z"}

def execute_falsification_experiment(
    experiment_spec: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cohort_mask: pd.Series
) -> float:
    """
    Execute the statistical test specified in experiment_spec against the Polish dataset.
    experiment_spec keys: experiment_name, null_sub_hypothesis, alternative_sub_hypothesis,
    statistical_test (must be in SUPPORTED_TESTS), columns_used.
    Returns p-value. Returns 1.0 on failure (conservative — contributes no evidence).
    Does not exec() any LLM-generated code.
    """
```

#### Notebook 02: LLN Experiment

**File:** `notebooks/02_llm_lln_experiment.ipynb`
**Question it answers:** Does collective LLM reasoning, with no data access, converge toward the empirical ground truth as the number of models queried increases? Which model families are closest?

**Structure:**
1. Markdown: LLN setup, cohort definition, ground truth μ
2. Load available Ollama models
3. Run borrower-level pairwise comparisons for all model pairs
4. Fit Bradley-Terry, validate stability
5. **Chart 1 — Pairwise comparison matrix:** Heatmap of win rates (model A vs model B). Title: *"Win rates are asymmetric across model families — open-source models diverge from Llama 3"*
6. **Chart 2 — LLN convergence curve:** Running BT-derived PD mean ± bootstrap CI vs k, vs LR estimate, vs ground truth. Title: *"LLM group mean converges toward empirical default rate after k=X models"* (or *"LLM group mean does not converge — high variance persists across k=N"*)
7. **Chart 3 — Model family decomposition:** Box plot of PD estimates by model family. Title: *"Within-family variance is lower than between-family variance — families encode different credit priors"*
8. **Chart 4 — Bias table:** For each model: estimated PD, bias vs ground truth, bias vs LR. Title: *"All models overestimate PD — collective LLM bias is systematic, not random"* (or the opposite)
9. Markdown: findings and interpretation. What does this tell us about using LLM consensus as a prior initialiser?

**Export:** `results/02_llm_lln_experiment.html`

---

#### Notebook 03: POPPER Falsification

**File:** `notebooks/03_popper_falsification.ipynb`
**Question it answers:** Can LLMs propose and execute valid statistical falsification experiments, and does the cumulative evidence achieve a statistically valid rejection decision?

**Structure:**
1. Markdown: POPPER setup, why e-values instead of Fisher, Type-I error guarantee
2. For each Ollama model, prompt for falsification experiment JSON spec
3. Parse and display proposed experiments in a formatted table
4. Execute each experiment via `execute_falsification_experiment()`
5. **Chart 1 — p-value trajectory:** p-values by round, coloured by model family. Title: *"P-values vary widely across rounds — some models propose strong tests, others propose weak ones"*
6. **Chart 2 — E-value trajectory:** Cumulative e-value (log scale) vs round, with 1/α rejection threshold. Title: *"Sufficient evidence to reject H0 at round X — cumulative E = Y"* (or "Insufficient evidence after N rounds")
7. **Chart 3 — LLN vs POPPER comparison table:** Rendered as a formatted table, not a chart. Side-by-side: LLN verdict, POPPER verdict, ground truth.
8. Markdown: which LLMs proposed the strongest falsification experiments? What does the failure mode analysis suggest about LLM statistical reasoning?

**Export:** `results/03_popper_falsification.html`

---

### ⚠️ CHECK IN after Stage 4 — Review LLM experiment findings before proceeding ⚠️

---

### STAGE 5 — Bayesian Analyst Layer + Production

**Goal:** Build the Bayesian prior update, FastAPI endpoint, and macro shift detector.

#### `src/analyst_sim.py`

```python
def simulate_category_comparisons(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_categories: dict,    # category → list of feature names
    cfg: dict,
    seed: int
) -> pd.DataFrame:
    """
    Simulate analyst pairwise comparisons at the feature category level.
    
    For each category, select cfg['analyst_sim']['n_companies'] companies.
    Generate cfg['analyst_sim']['n_analysts'] synthetic analysts, each with
    a calibration offset drawn from N(0, heterogeneity_sd).
    For each pair of companies, generate a comparison outcome probabilistically
    based on the true category-level risk difference + analyst noise.
    
    Returns: DataFrame with columns matching category_comparisons schema.
    Writes to SQLite.
    """
```

#### `src/prior_aggregation.py`

```python
def apply_forgetting_factor(
    comparisons: pd.DataFrame,
    reference_date: str,
    half_life_days: float
) -> pd.DataFrame:
    """
    Add weight column: weight = exp(-lambda * age_days).
    lambda = ln(2) / half_life_days.
    reference_date defaults to today.
    """

def compute_analyst_corrections(
    comparisons: pd.DataFrame,
    category: str
) -> dict:
    """
    Estimate each analyst's mean deviation from the global mean (fixed-effects intercept).
    Returns: {analyst_id: correction_value}
    """

def weighted_prior_stats(
    comparisons: pd.DataFrame,
    category: str
) -> dict:
    """
    After applying forgetting factor and analyst corrections, compute:
    mu_prior, sigma2_prior, precision_prior, n_observations, effective_n.
    Returns these as a dict — inputs to compute_posterior().
    """
```

#### `src/posterior_update.py`

```python
def bootstrap_coefficients(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int,
    seed: int
) -> pd.DataFrame:
    """
    Resample training data with replacement n_bootstrap times.
    Refit pipeline on each resample.
    Return DataFrame (n_bootstrap × n_features) of coefficient values.
    """

def compute_posterior(
    prior: dict,           # mu_prior, sigma2_prior
    mle_estimate: float,
    mle_variance: float,
    n_defaults: int,       # for thin-data safeguard
    cfg: dict
) -> dict:
    """
    Normal-Normal conjugate posterior.
    
    If n_defaults < cfg['analyst_sim']['min_defaults_for_mle']:
        tau_data = 0 (prior dominates; MLE not reliable)
    Else:
        tau_data = 1 / mle_variance
    
    Returns: mu_posterior, sigma2_posterior, prior_weight, data_weight.
    """

def update_all_coefficients(
    lr_pipeline: Pipeline,
    prior_stats_by_category: dict,
    feature_categories: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict
) -> dict:
    """
    For each feature category, compute posterior coefficient.
    Returns: {category: {mu_posterior, sigma2_posterior, prior_weight, data_weight}}
    """
```

#### `src/macro_shift.py`

```python
def detect_macro_shift(
    psi_report: pd.DataFrame,
    prior_history: pd.DataFrame,
    category: str,
    psi_threshold: float,
    prior_drift_sd_threshold: float,
    lookback_days: int
) -> dict:
    """
    Macro shift detector: fires when BOTH conditions are met:
    1. PSI for the feature category exceeds psi_threshold (default 0.2)
    2. The analyst prior mean for the category has moved > prior_drift_sd_threshold
       standard deviations from its value lookback_days ago
    
    A single condition alone is not sufficient evidence of a macro shift.
    
    Returns: {category, psi_value, prior_drift, is_macro_shift, triggered_at}
    """
```

#### Notebook 04: Bayesian Analyst Layer

**File:** `notebooks/04_bayesian_analyst_layer.ipynb`
**Question it answers:** Does the Bayesian posterior update improve calibration? How much does the analyst prior influence scores for thin-file vs data-rich borrowers?

**Structure:**
1. Markdown: setup, feature categories, analyst simulation design
2. Simulate category comparisons, fit Bradley-Terry per category
3. Apply forgetting factor and analyst corrections
4. Compute posterior coefficients
5. **Chart 1 — Prior vs MLE vs Posterior coefficients:** Per category, plot all three with uncertainty bands. Title: *"Prior dominates for sector risk (thin data); MLE dominates for financial health (data-rich)"*
6. **Chart 2 — Prior weight by category:** Bar chart of prior_weight per category. Title: *"Analyst prior contributes X% of sector risk coefficient but only Y% of financial health coefficient"*
7. **Chart 3 — Score distribution shift:** Histogram of PD scores before and after posterior update, overlaid. Title: *"Posterior update shifts scores upward for high-sector-risk companies"* (or whatever the data shows)
8. **Chart 4 — Macro shift detector simulation:** Timeline plot showing PSI and prior drift for a synthetic macro shift scenario. Title: *"Macro shift detector fires X days before PSI alone would trigger retraining"*
9. Markdown: what should a product manager take away from this? What signals indicate the system is working?
10. Demo API call: start FastAPI server in background, make a test request, display response

**Export:** `results/04_bayesian_analyst_layer.html`

---

### STAGE 6 — FastAPI, Tests, Final Polish

#### `app/main.py`

```python
# Load at startup (not per-request):
# - serialised LR Pipeline (latest from model registry)
# - posterior_coefficients from artefacts/

@app.on_event("startup")
async def load_models():
    """Load Pipeline and posterior coefficients. Log model version to console."""

@app.post("/score", response_model=ScoreResponse)
async def score_borrower(profile: BorrowerProfile) -> ScoreResponse:
    """
    Override LR coefficients with posterior means per feature category.
    Score the borrower.
    Return: predicted_pd, model_version, oot_gini, prior_weight_by_category.
    """

@app.get("/health")
async def health() -> dict:
    """Return: status, model_version, oot_gini."""

@app.get("/metadata")
async def metadata() -> dict:
    """Return model_metadata.json contents."""
```

#### `app/schemas.py`

Input schema: Pydantic model with one field per Polish dataset feature column. All fields typed as `float`, with `Field(description=...)` matching the Polish dataset documentation.

Output schema: `ScoreResponse` with `predicted_pd: float`, `model_version: str`, `oot_gini: float`, `prior_weight_by_category: dict[str, float]`.

#### `tests/test_api.py`

- Health check returns 200
- Score endpoint returns float between 0 and 1
- NaN input raises 422
- High-risk profile scores higher than low-risk profile

---

## Notebook Presentation Standards

These apply to every notebook without exception.

### Structure
Every notebook must have:
1. A title cell (H1) stating the question the notebook answers
2. A brief intro paragraph explaining why this question matters
3. Imports and config load (no magic numbers — `cfg = load_config()` for everything)
4. One markdown cell before each chart (what we're about to see and why)
5. One markdown cell after each chart (what we found and what it means)
6. A summary cell at the end with explicit findings and next steps

### Chart standards
- **Finding-first titles:** The chart title states the conclusion, not the description. "Default rate is highest in year 1" not "Default rate by year".
- **Consistent colour palette:** Use values from `config.yaml`. Blue (#2563eb) for primary, red (#dc2626) for risk/negative, green (#16a34a) for positive/reference.
- **All axes labelled with units**
- **Gridlines:** light grey, alpha=0.3
- **Figure size:** (12, 5) for single charts, (12, 8) for multi-panel
- **DPI:** 150 for saves
- **No untitled charts**
- **No charts without a markdown interpretation cell beneath them**

### Export requirement
Every notebook must export cleanly to HTML via:
```bash
jupyter nbconvert --to html --execute notebooks/{name}.ipynb --output results/{name}.html
```
This must work without manual cell execution. All file paths must be relative to the project root.

---

## src/ Module Standards

### Rules
- **One public function per module** (the rest are private helpers prefixed with `_`)
- **Exception:** `WoETransformer` is a class in `woe_transformer.py`; all modules that expose a class can have multiple class methods but only one class
- **Type hints** on all function signatures (arguments and return values)
- **Docstrings** on every public function: one-line summary, Args section, Returns section, Raises section (if applicable)
- **No magic numbers** — every threshold, bin count, learning rate, and hyperparameter is read from the `cfg` dict passed as an argument
- **No imports inside functions** (all imports at the top of the file)
- **No `print()` statements in src/ modules** — use `logging.getLogger(__name__)`

### Example module signature

```python
# src/monitoring.py

import logging
import numpy as np
import pandas as pd

from src.woe_transformer import WoETransformer

logger = logging.getLogger(__name__)


def monitor_all_features(
    transformer: WoETransformer,
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    cfg: dict
) -> pd.DataFrame:
    """
    Compute PSI for every numeric feature and classify distribution shift.
    
    Args:
        transformer: Fitted WoETransformer with bin_edges_ populated.
        X_train: Training set feature matrix (baseline distribution).
        X_current: Current cohort feature matrix.
        cfg: Configuration dict; reads cfg['psi']['stable_threshold'],
             cfg['psi']['monitor_threshold'], cfg['psi']['epsilon'].
    
    Returns:
        pd.DataFrame with columns: feature, psi, status.
        Status values: 'STABLE' | 'MONITOR' | 'RETRAIN'.
        Sorted by psi descending.
    
    Raises:
        ValueError: If X_current contains features not in transformer.feature_names_.
    """
    ...
```

---

## Testing Standards

- `pytest` with no magic numbers in test files — fixtures load `config.yaml`
- Every test is named `test_{what_it_tests}` — no `test_1`, `test_a`
- All 5 WoE transformer tests must pass before any modelling code is built
- API tests use `httpx.AsyncClient` with `app` as the ASGI app (no running server required)
- Target: all tests pass with `make test` from project root

---

## decisions.md

As you build, maintain `decisions.md` in the project root. One entry per major design decision, written in first person. Template:

```markdown
## [Date] — [Decision title]

**Decision:** [What was decided]
**Why:** [The specific reason — reference the config key, the test output, or the finding that motivated it]
**Alternatives considered:** [What else was considered and why it was rejected]
**Confidence:** [High / Medium / Low — honest about uncertainty]
```

Examples to include:
- Why out-of-time split rather than random
- Why logistic regression as the base for the Bayesian update (not XGBoost)
- Why Bradley-Terry rather than direct numeric scores
- Why Normal-Normal conjugate rather than full MCMC
- Why isotonic calibration for XGBoost
- Why e-values rather than Fisher's combined test

---

## Colour Palette Reference

| Use | Hex | Name |
|---|---|---|
| Primary / non-default / neutral | #2563eb | Blue |
| Risk / default / negative | #dc2626 | Red |
| Positive / reference / ground truth | #16a34a | Green |
| LR estimate | #16a34a | Green |
| LLM mean | #2563eb | Blue |
| Ground truth | #dc2626 | Red |
| Neutral / secondary | #6b7280 | Grey |
| Background | #f8fafc | Off-white |

---

## Final Checklist Before Submitting

- [ ] `make install && make download && make explore` runs without error
- [ ] `make test` passes (all tests green)
- [ ] `make html` exports all 5 notebooks to `results/`
- [ ] `make api` starts FastAPI; `/health` returns 200; `/score` returns a PD for a test profile
- [ ] All charts have finding-first titles
- [ ] All notebooks have markdown narrative before and after every chart
- [ ] `decisions.md` has entries for all major design choices
- [ ] No magic numbers anywhere in src/ or app/
- [ ] `model_metadata.json` exists in `artefacts/` with non-null OOT metrics
- [ ] `retraining_log.jsonl` exists with at least one entry
