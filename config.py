from pathlib import Path

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR   = BASE_DIR / "data"

CUSTOMER_ID_COL  = "UniqueID"
TARGET           = "next_3m_txn_count"
DATE_COL         = "TransactionDate"
AMOUNT_COL       = "TransactionAmount"
AMOUNT_ABS_COL   = "amount_abs"
DEBIT_CREDIT_COL = "IsDebitCredit"
CHANNEL_COL      = "TransactionBatchDescription"
TXN_TYPE_COL     = "TransactionTypeDescription"
CATEGORY_COL     = "TransactionTypeDescription"
BALANCE_COL      = "StatementBalance"

TRAIN_CUTOFF = "2015-11-01"

SA_HOLIDAY_WINDOWS = [
    ("12-01", "01-15"),
    ("03-21", "03-21"),
    ("04-27", "04-27"),
    ("05-01", "05-01"),
    ("06-16", "06-16"),
    ("08-09", "08-09"),
    ("09-24", "09-24"),
    ("12-16", "12-16"),
    ("12-25", "12-26"),
]

ROLLING_WINDOWS       = [1, 3, 6, 12]
LAG_MONTHS            = [1, 2, 3, 6, 12]
VELOCITY_WINDOWS      = [7, 14, 30, 60, 90]
DECAY_HALF_LIVES      = [30, 90, 180]
N_CLUSTERS            = 5
N_PCA_COMPONENTS      = 10
N_TYPE_SVD_COMPONENTS = 10
ISO_CONTAMINATION     = 0.05
ISO_N_ESTIMATORS      = 100

RANDOM_SEED        = 42
N_FOLDS            = 5
N_CV_FOLDS         = 5
MIN_TRAIN_FRAC     = 0.5
OPTUNA_N_TRIALS    = 50
LGB_EARLY_STOPPING = 100
OPTUNA_TIMEOUT     = 600
OPTUNA_DIRECTION   = "minimize"
ENSEMBLE_WEIGHTS      = {"xgb": 0.5, "lgb": 0.5}
ZERO_STAGE_THRESHOLD  = 0.10
USE_POISSON_OBJECTIVE = True

XGB_PARAMS = dict(
    objective        = "count:poisson",
    eval_metric      = "poisson-nloglik",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 3,
    subsample        = 0.80,
    colsample_bytree = 0.70,
    reg_alpha        = 0.05,
    reg_lambda       = 1.50,
    tree_method      = "hist",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

LGB_PARAMS = dict(
    objective         = "poisson",
    metric            = "poisson",
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 63,
    min_child_samples = 20,
    subsample         = 0.80,
    colsample_bytree  = 0.70,
    reg_alpha         = 0.05,
    reg_lambda        = 1.50,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)

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
    random_state          = RANDOM_SEED,
    n_jobs                = -1,
    early_stopping_rounds = 100,
)

XGB_POISSON_OVERRIDES = dict(
    objective   = "count:poisson",
    eval_metric = "poisson-nloglik",
)
LGB_POISSON_OVERRIDES = dict(
    objective = "poisson",
    metric    = "poisson",
)