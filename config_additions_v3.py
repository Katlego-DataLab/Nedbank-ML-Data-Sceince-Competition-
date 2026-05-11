"""
config_additions_v3.py
======================
Paste these additions into your existing config.py.
They add the new constants needed by feature_engineering_v3.py
and modelling_v3.py without changing anything else.
"""

# ---------------------------------------------------------------------------
# Paste these lines into config.py alongside the existing constants
# ---------------------------------------------------------------------------

# Velocity windows (days)
VELOCITY_WINDOWS = [7, 14, 30, 60, 90]

# Exponential decay half-lives (days) for recency-weighted features
DECAY_HALF_LIVES = [30, 90, 180]

# SVD components for transaction-type behavioral embeddings
N_TYPE_SVD_COMPONENTS = 10

# Two-stage zero-inflation — minimum probability to flag as "active"
ZERO_STAGE_THRESHOLD = 0.10   # predictions below this P(active) → set to 0

# Poisson objective flags
USE_POISSON_OBJECTIVE = True

# XGBoost Poisson override  (merged with XGB_PARAMS at runtime in modelling_v3)
XGB_POISSON_OVERRIDES = dict(
    objective  = "count:poisson",
    eval_metric= "poisson-nloglik",
)

# LightGBM Poisson override
LGB_POISSON_OVERRIDES = dict(
    objective = "poisson",
    metric    = "poisson",
)

# Stage-1 classifier params (binary XGBoost)
XGB_STAGE1_PARAMS = dict(
    objective             = "binary:logistic",
    eval_metric           = "logloss",
    n_estimators          = 800,
    learning_rate         = 0.05,
    max_depth             = 6,
    min_child_weight      = 3,
    subsample             = 0.80,
    colsample_bytree      = 0.70,
    reg_alpha             = 0.05,
    reg_lambda            = 1.50,
    tree_method           = "hist",
    random_state          = 42,   # use RANDOM_SEED constant when pasting
    n_jobs                = -1,
    early_stopping_rounds = 100,
)
