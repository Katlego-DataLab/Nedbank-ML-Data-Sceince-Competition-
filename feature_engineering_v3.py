"""
feature_engineering_v3.py  — Drop-in replacement for feature_engineering.py

Key additions vs v2:
  1.  Holiday-season granular features  (Nov/Dec/Jan per-year, spike ratios)
  2.  Full debit/credit decomposition   (stats + ratios per direction)
  3.  Balance trajectory features       (volatility, overdraft, EOM balance)
  4.  Transaction-type behavioral embeddings (TF-IDF-style freq vectors + clustering)
  5.  Customer velocity features        (7d / 14d / 30d / 60d / 90d windows)
  6.  Recency-weighted (exponential decay) aggregations
  7.  Customer consistency features     (CV, streaks, Gini-activity)
  All existing v2 features are preserved unchanged.
"""
import gc
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from config import (
    CUSTOMER_ID_COL,
    DATE_COL,
    AMOUNT_ABS_COL,
    AMOUNT_COL,
    CHANNEL_COL,
    CATEGORY_COL,
    TRAIN_CUTOFF,
    SA_HOLIDAY_WINDOWS,
    ROLLING_WINDOWS,
    LAG_MONTHS,
    N_CLUSTERS,
    N_PCA_COMPONENTS,
    ISO_CONTAMINATION,
    ISO_N_ESTIMATORS,
    RANDOM_SEED,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("nedbank.features")

CUTOFF = pd.Timestamp(TRAIN_CUTOFF)

# ---------------------------------------------------------------------------
# Decay constants for recency-weighted features (half-life in days)
# ---------------------------------------------------------------------------
DECAY_HALF_LIVES = [30, 90, 180]   # short / medium / long memory


# ===========================================================================
# Helpers (unchanged from v2)
# ===========================================================================

def _is_sa_holiday(dt):
    if pd.isna(dt):
        return 0
    mmdd = dt.strftime("%m-%d")
    for start, end in SA_HOLIDAY_WINDOWS:
        if start > end:
            if mmdd >= start or mmdd <= end:
                return 1
        else:
            if start <= mmdd <= end:
                return 1
    return 0


def _numeric_cols(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c != CUSTOMER_ID_COL]


def _debit_sum(x):
    return (x == "debit").sum()


def _credit_sum(x):
    return (x == "credit").sum()


# ===========================================================================
# 1. Temporal features  (unchanged)
# ===========================================================================

def build_temporal_features(txn):
    log.info("Building temporal features ...")
    d = txn[DATE_COL]
    txn["txn_year"]           = d.dt.year.astype(np.int16)
    txn["txn_month"]          = d.dt.month.astype(np.int8)
    txn["txn_quarter"]        = d.dt.quarter.astype(np.int8)
    txn["txn_dayofweek"]      = d.dt.dayofweek.astype(np.int8)
    txn["txn_is_weekend"]     = (d.dt.dayofweek >= 5).astype(np.int8)
    txn["txn_is_month_start"] = d.dt.is_month_start.astype(np.int8)
    txn["txn_is_month_end"]   = d.dt.is_month_end.astype(np.int8)
    txn["txn_is_sa_holiday"]  = d.apply(_is_sa_holiday).astype(np.int8)
    txn["txn_month_sin"]      = np.sin(2 * np.pi * txn["txn_month"] / 12).astype(np.float32)
    txn["txn_month_cos"]      = np.cos(2 * np.pi * txn["txn_month"] / 12).astype(np.float32)
    txn["txn_quarter_sin"]    = np.sin(2 * np.pi * txn["txn_quarter"] / 4).astype(np.float32)
    txn["txn_quarter_cos"]    = np.cos(2 * np.pi * txn["txn_quarter"] / 4).astype(np.float32)
    start_date = pd.Timestamp("2012-12-01")
    txn["days_since_start"]   = (d - start_date).dt.days.astype(np.int16)
    txn["ym"]                 = d.dt.to_period("M")
    # Day-of-month (useful for end-of-month salary/payment detection)
    txn["txn_day"]            = d.dt.day.astype(np.int8)
    txn["days_to_month_end"]  = (
        d.dt.days_in_month - d.dt.day
    ).astype(np.int8)
    return txn


# ===========================================================================
# 2. Monthly table  (extended with debit/credit amount stats)
# ===========================================================================

def build_monthly_table(txn):
    log.info("  Aggregating transactions to monthly table ...")

    def _debit_amt(sub):
        mask = sub["debit_credit"] == "debit"
        vals = sub.loc[mask, AMOUNT_ABS_COL]
        return pd.Series({
            "monthly_debit_count":  mask.sum(),
            "monthly_debit_sum":    vals.sum() if len(vals) else 0.0,
            "monthly_debit_mean":   vals.mean() if len(vals) else 0.0,
            "monthly_debit_std":    vals.std()  if len(vals) > 1 else 0.0,
            "monthly_credit_count": (~mask).sum(),
            "monthly_credit_sum":   sub.loc[~mask, AMOUNT_ABS_COL].sum(),
            "monthly_credit_mean":  sub.loc[~mask, AMOUNT_ABS_COL].mean() if (~mask).any() else 0.0,
            "monthly_credit_std":   sub.loc[~mask, AMOUNT_ABS_COL].std()  if (~mask).sum() > 1 else 0.0,
        })

    # Fast path: pre-split debit/credit columns
    if "debit_credit" in txn.columns:
        is_debit   = txn["debit_credit"] == "debit"
        txn["_debit_amt"]  = np.where(is_debit,  txn[AMOUNT_ABS_COL], 0.0)
        txn["_credit_amt"] = np.where(~is_debit, txn[AMOUNT_ABS_COL], 0.0)
        txn["_is_debit"]   = is_debit.astype(np.int8)
        txn["_is_credit"]  = (~is_debit).astype(np.int8)

    monthly = (
        txn.groupby([CUSTOMER_ID_COL, "ym"], observed=True)
        .agg(
            monthly_txn_count    =(CUSTOMER_ID_COL,      "count"),
            monthly_total_amt    =(AMOUNT_ABS_COL,        "sum"),
            monthly_mean_amt     =(AMOUNT_ABS_COL,        "mean"),
            monthly_max_amt      =(AMOUNT_ABS_COL,        "max"),
            monthly_min_amt      =(AMOUNT_ABS_COL,        "min"),
            monthly_std_amt      =(AMOUNT_ABS_COL,        "std"),
            monthly_debit_cnt    =("debit_credit",        _debit_sum),
            monthly_credit_cnt   =("debit_credit",        _credit_sum),
            monthly_weekend_cnt  =("txn_is_weekend",      "sum"),
            monthly_holiday_cnt  =("txn_is_sa_holiday",   "sum"),
            # new debit/credit amount columns
            monthly_debit_sum    =("_debit_amt",          "sum"),
            monthly_credit_sum   =("_credit_amt",         "sum"),
            monthly_debit_mean   =("_debit_amt",          lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            monthly_credit_mean  =("_credit_amt",         lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        )
        .reset_index()
    )

    monthly["debit_credit_ratio"]         = np.where(
        monthly["monthly_credit_cnt"] > 0,
        monthly["monthly_debit_cnt"] / monthly["monthly_credit_cnt"],
        np.nan,
    ).astype(np.float32)

    monthly["debit_credit_amount_ratio"]  = np.where(
        monthly["monthly_credit_sum"] > 0,
        monthly["monthly_debit_sum"] / monthly["monthly_credit_sum"],
        np.nan,
    ).astype(np.float32)

    monthly["weekend_txn_frac"]           = (
        monthly["monthly_weekend_cnt"] / (monthly["monthly_txn_count"] + 1e-9)
    ).astype(np.float32)

    monthly["holiday_txn_frac_monthly"]   = (
        monthly["monthly_holiday_cnt"] / (monthly["monthly_txn_count"] + 1e-9)
    ).astype(np.float32)

    monthly["amt_cv"] = (
        monthly["monthly_std_amt"] / (monthly["monthly_mean_amt"] + 1e-9)
    ).astype(np.float32)

    monthly["net_flow"] = (
        monthly["monthly_credit_sum"] - monthly["monthly_debit_sum"]
    ).astype(np.float32)

    for col in monthly.select_dtypes(include=["float64"]).columns:
        monthly[col] = monthly[col].astype(np.float32)
    for col in monthly.select_dtypes(include=["int64"]).columns:
        monthly[col] = monthly[col].astype(np.int32)

    # Drop temp columns from txn (inplace)
    txn.drop(columns=["_debit_amt", "_credit_amt", "_is_debit", "_is_credit"],
             inplace=True, errors="ignore")

    log.info(f"  Monthly table: {monthly.shape}")
    return monthly


# ===========================================================================
# 3. Customer aggregates  (extended with holiday, debit/credit, consistency)
# ===========================================================================

def build_customer_aggregates(txn):
    log.info("Building customer aggregate features ...")

    monthly = build_monthly_table(txn)
    del txn
    gc.collect()

    # -----------------------------------------------------------------------
    # 3a. Full-history aggregates (v2 baseline)
    # -----------------------------------------------------------------------
    features = (
        monthly.groupby(CUSTOMER_ID_COL)
        .agg(
            hist_months_active       =("ym",                       "nunique"),
            hist_total_txn           =("monthly_txn_count",        "sum"),
            hist_mean_monthly_txn    =("monthly_txn_count",        "mean"),
            hist_std_monthly_txn     =("monthly_txn_count",        "std"),
            hist_median_monthly_txn  =("monthly_txn_count",        "median"),
            hist_max_monthly_txn     =("monthly_txn_count",        "max"),
            hist_min_monthly_txn     =("monthly_txn_count",        "min"),
            hist_q75_monthly_txn     =("monthly_txn_count",        lambda x: x.quantile(0.75)),
            hist_q25_monthly_txn     =("monthly_txn_count",        lambda x: x.quantile(0.25)),
            hist_total_amount        =("monthly_total_amt",        "sum"),
            hist_mean_amount         =("monthly_total_amt",        "mean"),
            hist_std_amount          =("monthly_total_amt",        "std"),
            hist_max_amount          =("monthly_total_amt",        "max"),
            hist_mean_debit_ratio    =("debit_credit_ratio",       "mean"),
            hist_std_debit_ratio     =("debit_credit_ratio",       "std"),
            hist_mean_weekend_frac   =("weekend_txn_frac",         "mean"),
            hist_mean_holiday_frac   =("holiday_txn_frac_monthly", "mean"),
            hist_mean_amt_cv         =("amt_cv",                   "mean"),
            hist_zero_months         =("monthly_txn_count",        lambda x: (x == 0).sum()),
            # new: debit/credit amount history
            hist_total_debit_sum     =("monthly_debit_sum",        "sum"),
            hist_total_credit_sum    =("monthly_credit_sum",       "sum"),
            hist_mean_debit_sum      =("monthly_debit_sum",        "mean"),
            hist_mean_credit_sum     =("monthly_credit_sum",       "mean"),
            hist_mean_net_flow       =("net_flow",                 "mean"),
            hist_std_net_flow        =("net_flow",                 "std"),
        )
        .reset_index()
    )

    features["hist_cv_txn"] = (
        features["hist_std_monthly_txn"] / (features["hist_mean_monthly_txn"] + 1e-9)
    ).astype(np.float32)

    features["hist_iqr_txn"] = (
        features["hist_q75_monthly_txn"] - features["hist_q25_monthly_txn"]
    ).astype(np.float32)

    features["hist_activity_rate"] = (
        (features["hist_months_active"] - features["hist_zero_months"])
        / (features["hist_months_active"] + 1e-9)
    ).astype(np.float32)

    features["hist_amount_per_txn"] = (
        features["hist_total_amount"] / (features["hist_total_txn"] + 1e-9)
    ).astype(np.float32)

    features["hist_debit_credit_amount_ratio"] = (
        features["hist_total_debit_sum"] / (features["hist_total_credit_sum"] + 1e-9)
    ).astype(np.float32)

    for col in features.select_dtypes(include=["float64"]).columns:
        features[col] = features[col].astype(np.float32)
    for col in features.select_dtypes(include=["int64"]).columns:
        features[col] = features[col].astype(np.int32)

    # -----------------------------------------------------------------------
    # 3b. Rolling window aggregates (v2 baseline, extended)
    # -----------------------------------------------------------------------
    cutoff_period = CUTOFF.to_period("M")
    for n in ROLLING_WINDOWS:
        start_period = (CUTOFF - pd.DateOffset(months=n)).to_period("M")
        window = monthly[
            (monthly["ym"] >= start_period) & (monthly["ym"] <= cutoff_period)
        ]
        roll = (
            window.groupby(CUSTOMER_ID_COL)
            .agg(**{
                f"roll{n}m_txn_count":     ("monthly_txn_count",         "sum"),
                f"roll{n}m_mean_txn":      ("monthly_txn_count",         "mean"),
                f"roll{n}m_std_txn":       ("monthly_txn_count",         "std"),
                f"roll{n}m_max_txn":       ("monthly_txn_count",         "max"),
                f"roll{n}m_total_amt":     ("monthly_total_amt",         "sum"),
                f"roll{n}m_mean_amt":      ("monthly_total_amt",         "mean"),
                f"roll{n}m_debit_ratio":   ("debit_credit_ratio",        "mean"),
                f"roll{n}m_weekend_frac":  ("weekend_txn_frac",          "mean"),
                # new
                f"roll{n}m_debit_sum":     ("monthly_debit_sum",         "sum"),
                f"roll{n}m_credit_sum":    ("monthly_credit_sum",        "sum"),
                f"roll{n}m_net_flow":      ("net_flow",                  "sum"),
                f"roll{n}m_debit_cnt":     ("monthly_debit_cnt",         "sum"),
                f"roll{n}m_credit_cnt":    ("monthly_credit_cnt",        "sum"),
            })
            .reset_index()
        )
        roll[f"roll{n}m_dc_amt_ratio"] = (
            roll[f"roll{n}m_debit_sum"] / (roll[f"roll{n}m_credit_sum"] + 1e-9)
        ).astype(np.float32)
        for col in roll.select_dtypes(include=["float64"]).columns:
            roll[col] = roll[col].astype(np.float32)
        features = features.merge(roll, on=CUSTOMER_ID_COL, how="left")
        del roll, window
        gc.collect()

    # -----------------------------------------------------------------------
    # 3c. Lag features (v2 baseline)
    # -----------------------------------------------------------------------
    lag_frames = []
    for lag in LAG_MONTHS:
        period = (CUTOFF - pd.DateOffset(months=lag - 1)).to_period("M")
        lag_df = (
            monthly[monthly["ym"] == period]
            [[CUSTOMER_ID_COL, "monthly_txn_count", "monthly_total_amt",
              "debit_credit_ratio", "monthly_mean_amt",
              "monthly_debit_sum", "monthly_credit_sum", "net_flow"]]
            .rename(columns={
                "monthly_txn_count":   f"lag{lag}m_txn",
                "monthly_total_amt":   f"lag{lag}m_amt",
                "debit_credit_ratio":  f"lag{lag}m_debit_ratio",
                "monthly_mean_amt":    f"lag{lag}m_mean_amt",
                "monthly_debit_sum":   f"lag{lag}m_debit_sum",
                "monthly_credit_sum":  f"lag{lag}m_credit_sum",
                "net_flow":            f"lag{lag}m_net_flow",
            })
        )
        for col in lag_df.select_dtypes(include=["float64"]).columns:
            lag_df[col] = lag_df[col].astype(np.float32)
        lag_frames.append(lag_df)
        features = features.merge(lag_df, on=CUSTOMER_ID_COL, how="left")

    # MoM growth (v2)
    if len(lag_frames) >= 2:
        mom = lag_frames[0].merge(lag_frames[1], on=CUSTOMER_ID_COL, how="outer")
        mom["mom_txn_growth"] = (
            (mom["lag1m_txn"] - mom["lag2m_txn"])
            / (mom["lag2m_txn"].replace(0, np.nan))
        ).astype(np.float32)
        mom["mom_amt_growth"] = (
            (mom["lag1m_amt"] - mom["lag2m_amt"])
            / (mom["lag2m_amt"].replace(0, np.nan))
        ).astype(np.float32)
        features = features.merge(
            mom[[CUSTOMER_ID_COL, "mom_txn_growth", "mom_amt_growth"]],
            on=CUSTOMER_ID_COL, how="left"
        )
        del mom

    if "roll3m_txn_count" in features.columns and "roll6m_txn_count" in features.columns:
        features["growth_3m_vs_6m"] = (
            features["roll3m_txn_count"]
            / (features["roll6m_txn_count"].replace(0, np.nan) / 2)
        ).astype(np.float32)

    if "roll1m_txn_count" in features.columns and "roll3m_mean_txn" in features.columns:
        features["trend_1m_vs_3m"] = (
            features["roll1m_txn_count"]
            / (features["roll3m_mean_txn"].replace(0, np.nan))
        ).astype(np.float32)

    if len(lag_frames) >= 3:
        acc = lag_frames[0].merge(lag_frames[1], on=CUSTOMER_ID_COL, how="outer")
        acc = acc.merge(lag_frames[2], on=CUSTOMER_ID_COL, how="outer")
        acc["txn_acceleration"] = (
            (acc["lag1m_txn"] - acc["lag2m_txn"])
            - (acc["lag2m_txn"] - acc["lag3m_txn"])
        ).astype(np.float32)
        features = features.merge(
            acc[[CUSTOMER_ID_COL, "txn_acceleration"]], on=CUSTOMER_ID_COL, how="left"
        )
        del acc

    # -----------------------------------------------------------------------
    # 3d. [NEW] Granular holiday-season features
    # -----------------------------------------------------------------------
    log.info("  Building granular holiday-season features ...")

    # Yearly totals (denominator for ratios)
    monthly["year_int"] = monthly["ym"].apply(lambda p: p.year)
    yearly_txn = (
        monthly.groupby([CUSTOMER_ID_COL, "year_int"])["monthly_txn_count"]
        .sum()
        .rename("yearly_txn")
        .reset_index()
    )
    # mean yearly count per customer
    yearly_mean = (
        yearly_txn.groupby(CUSTOMER_ID_COL)["yearly_txn"]
        .mean()
        .rename("hist_mean_yearly_txn")
        .reset_index()
    )
    features = features.merge(yearly_mean, on=CUSTOMER_ID_COL, how="left")
    del yearly_mean

    monthly["month_int"] = monthly["ym"].apply(lambda p: p.month)

    def _month_features(month_num, label):
        m_data = monthly[monthly["month_int"] == month_num]
        if len(m_data) == 0:
            return None
        agg = (
            m_data.groupby(CUSTOMER_ID_COL)
            .agg(**{
                f"hist_{label}_mean_txn":   ("monthly_txn_count", "mean"),
                f"hist_{label}_std_txn":    ("monthly_txn_count", "std"),
                f"hist_{label}_max_txn":    ("monthly_txn_count", "max"),
                f"hist_{label}_mean_amt":   ("monthly_total_amt", "mean"),
                f"hist_{label}_mean_debit": ("monthly_debit_sum", "mean"),
                f"hist_{label}_mean_credit":("monthly_credit_sum","mean"),
            })
            .reset_index()
        )
        return agg

    for month_num, label in [(11, "nov"), (12, "dec"), (1, "jan")]:
        agg = _month_features(month_num, label)
        if agg is not None:
            features = features.merge(agg, on=CUSTOMER_ID_COL, how="left")
            del agg

    # Dec spike ratio: dec_txn / mean_other_months
    if "hist_dec_mean_txn" in features.columns:
        features["dec_spike_ratio"] = (
            features["hist_dec_mean_txn"]
            / (features["hist_mean_monthly_txn"] + 1e-9)
        ).astype(np.float32)

        # Dec vs yearly
        features["dec_yearly_ratio"] = (
            features["hist_dec_mean_txn"] * 12
            / (features["hist_mean_yearly_txn"] + 1e-9)
        ).astype(np.float32)

    # Nov-to-Dec ramp ratio (signals holiday ramp-up speed)
    if "hist_nov_mean_txn" in features.columns and "hist_dec_mean_txn" in features.columns:
        features["nov_dec_ratio"] = (
            features["hist_dec_mean_txn"]
            / (features["hist_nov_mean_txn"] + 1e-9)
        ).astype(np.float32)

    # Festive (Dec+Jan) features (v2 baseline + new amounts)
    festive = monthly[monthly["month_int"].isin([12, 1])]
    if len(festive) > 0:
        festive_agg = (
            festive.groupby(CUSTOMER_ID_COL)
            .agg(
                festive_mean_txn=("monthly_txn_count",  "mean"),
                festive_total_txn=("monthly_txn_count", "sum"),
                festive_mean_amt=("monthly_total_amt",  "mean"),
            )
            .reset_index()
        )
        features = features.merge(festive_agg, on=CUSTOMER_ID_COL, how="left")
        features[["festive_mean_txn", "festive_total_txn",
                   "festive_mean_amt"]] = features[
            ["festive_mean_txn", "festive_total_txn", "festive_mean_amt"]
        ].fillna(0)
        del festive_agg

    # Nov specifically (v2 baseline label)
    nov_data = monthly[monthly["month_int"] == 11]
    if len(nov_data) > 0:
        nov_agg = (
            nov_data.groupby(CUSTOMER_ID_COL)["monthly_txn_count"]
            .mean()
            .reset_index(name="hist_nov_mean_txn")
        )
        # Only add if not already from _month_features
        if "hist_nov_mean_txn" not in features.columns:
            features = features.merge(nov_agg, on=CUSTOMER_ID_COL, how="left")
            features["hist_nov_mean_txn"].fillna(0, inplace=True)
        del nov_agg

    # Q4 aggregate (Oct–Dec) — the quarter that immediately precedes prediction
    q4 = monthly[monthly["month_int"].isin([10, 11, 12])]
    if len(q4) > 0:
        q4_agg = (
            q4.groupby(CUSTOMER_ID_COL)
            .agg(
                hist_q4_mean_txn=("monthly_txn_count", "mean"),
                hist_q4_total_txn=("monthly_txn_count","sum"),
            )
            .reset_index()
        )
        features = features.merge(q4_agg, on=CUSTOMER_ID_COL, how="left")
        features[["hist_q4_mean_txn", "hist_q4_total_txn"]] = features[
            ["hist_q4_mean_txn", "hist_q4_total_txn"]
        ].fillna(0)
        del q4_agg

    # -----------------------------------------------------------------------
    # 3e. [NEW] Customer consistency / streak features
    # -----------------------------------------------------------------------
    log.info("  Building consistency/streak features ...")

    def _streaks(counts):
        """Longest active and inactive streak in monthly counts."""
        active   = longest_active = cur_active = 0
        inactive = longest_inactive = cur_inactive = 0
        for c in counts:
            if c > 0:
                cur_active  += 1
                cur_inactive = 0
            else:
                cur_inactive += 1
                cur_active   = 0
            longest_active   = max(longest_active, cur_active)
            longest_inactive = max(longest_inactive, cur_inactive)
        return longest_active, longest_inactive

    def _gini(counts):
        """Gini coefficient of activity distribution."""
        arr = np.sort(np.abs(counts))
        n   = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float(((2 * idx - n - 1) * arr).sum() / (n * arr.sum()))

    streak_data = []
    for cust_id, grp in monthly.sort_values("ym").groupby(CUSTOMER_ID_COL):
        counts   = grp["monthly_txn_count"].values
        la, li   = _streaks(counts)
        gini_val = _gini(counts)
        streak_data.append({
            CUSTOMER_ID_COL:          cust_id,
            "streak_longest_active":  la,
            "streak_longest_inactive":li,
            "consistency_gini":       gini_val,
        })
    streak_df = pd.DataFrame(streak_data)
    for col in streak_df.select_dtypes(include=["float64"]).columns:
        streak_df[col] = streak_df[col].astype(np.float32)
    features = features.merge(streak_df, on=CUSTOMER_ID_COL, how="left")
    del streak_data, streak_df

    # Coefficient of variation (monthly txn) — high CV = irregular customer
    if "hist_std_monthly_txn" in features.columns:
        features["hist_txn_cv"] = (
            features["hist_std_monthly_txn"]
            / (features["hist_mean_monthly_txn"] + 1e-9)
        ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    del monthly, festive, nov_data, q4, lag_frames
    gc.collect()

    for col in features.select_dtypes(include=["float64"]).columns:
        features[col] = features[col].astype(np.float32)
    for col in features.select_dtypes(include=["int64"]).columns:
        features[col] = features[col].astype(np.int32)

    log.info(f"  Customer aggregate features: {features.shape}")
    return features


# ===========================================================================
# 4. [NEW] Customer velocity features  (7 / 14 / 30 / 60 / 90 day windows)
# ===========================================================================

def build_velocity_features(txn):
    """
    Fine-grained short-to-medium-term activity based on calendar days
    before cutoff.  These capture momentum much better than monthly windows
    at 7, 14, 30, 60, 90 days.
    """
    log.info("Building velocity features ...")

    windows = {
        "7d":  7,
        "14d": 14,
        "30d": 30,
        "60d": 60,
        "90d": 90,
    }

    result_frames = []
    for label, days in windows.items():
        start = CUTOFF - pd.Timedelta(days=days)
        mask  = (txn[DATE_COL] > start) & (txn[DATE_COL] <= CUTOFF)
        sub   = txn[mask]

        if len(sub) == 0:
            continue

        is_debit = sub["debit_credit"] == "debit" if "debit_credit" in sub.columns else pd.Series(False, index=sub.index)

        agg = (
            sub.groupby(CUSTOMER_ID_COL)
            .apply(lambda g: pd.Series({
                f"vel_{label}_txn_count":    len(g),
                f"vel_{label}_total_amt":    g[AMOUNT_ABS_COL].sum(),
                f"vel_{label}_mean_amt":     g[AMOUNT_ABS_COL].mean(),
                f"vel_{label}_debit_count":  (g["debit_credit"] == "debit").sum() if "debit_credit" in g.columns else 0,
                f"vel_{label}_credit_count": (g["debit_credit"] == "credit").sum() if "debit_credit" in g.columns else 0,
            }))
            .reset_index()
        )
        for col in agg.select_dtypes(include=["float64"]).columns:
            agg[col] = agg[col].astype(np.float32)
        result_frames.append(agg)
        del sub, agg
        gc.collect()

    if not result_frames:
        return pd.DataFrame({CUSTOMER_ID_COL: txn[CUSTOMER_ID_COL].unique()})

    result = result_frames[0]
    for f in result_frames[1:]:
        result = result.merge(f, on=CUSTOMER_ID_COL, how="outer")
    result.fillna(0, inplace=True)

    # Velocity ratios — the key predictive signal
    if "vel_30d_txn_count" in result.columns and "vel_90d_txn_count" in result.columns:
        result["vel_ratio_30_90"] = (
            result["vel_30d_txn_count"] / (result["vel_90d_txn_count"] / 3 + 1e-9)
        ).astype(np.float32)  # >1 = accelerating, <1 = decelerating

    if "vel_7d_txn_count" in result.columns and "vel_30d_txn_count" in result.columns:
        result["vel_ratio_7_30"] = (
            result["vel_7d_txn_count"] / (result["vel_30d_txn_count"] / 4.3 + 1e-9)
        ).astype(np.float32)

    if "vel_14d_txn_count" in result.columns and "vel_60d_txn_count" in result.columns:
        result["vel_ratio_14_60"] = (
            result["vel_14d_txn_count"] / (result["vel_60d_txn_count"] / 4.3 + 1e-9)
        ).astype(np.float32)

    log.info(f"  Velocity features: {result.shape}")
    return result


# ===========================================================================
# 5. [NEW] Recency-weighted (exponential decay) aggregations
# ===========================================================================

def build_decay_weighted_features(txn):
    """
    Each transaction's contribution is weighted by exp(-days_ago / half_life).
    Three half-lives: 30d (short), 90d (medium), 180d (long).
    Features: weighted count, weighted amount sum, weighted debit/credit counts.
    """
    log.info("Building exponential-decay weighted features ...")

    txn = txn[txn[DATE_COL] <= CUTOFF].copy()
    txn["days_ago"] = (CUTOFF - txn[DATE_COL]).dt.days.astype(np.float32)

    rows = []
    for hl in DECAY_HALF_LIVES:
        decay_rate   = np.log(2) / hl
        txn["_w"]    = np.exp(-decay_rate * txn["days_ago"]).astype(np.float32)
        txn["_wamt"] = (txn["_w"] * txn[AMOUNT_ABS_COL]).astype(np.float32)

        is_debit = (txn["debit_credit"] == "debit") if "debit_credit" in txn.columns else pd.Series(False, index=txn.index)
        txn["_wdebit"]  = (txn["_w"] * is_debit.astype(np.float32)).astype(np.float32)
        txn["_wcredit"] = (txn["_w"] * (~is_debit).astype(np.float32)).astype(np.float32)

        agg = (
            txn.groupby(CUSTOMER_ID_COL)
            .agg(**{
                f"decay{hl}d_w_count":  ("_w",       "sum"),
                f"decay{hl}d_w_amt":    ("_wamt",    "sum"),
                f"decay{hl}d_w_debit":  ("_wdebit",  "sum"),
                f"decay{hl}d_w_credit": ("_wcredit", "sum"),
            })
            .reset_index()
        )
        agg[f"decay{hl}d_dc_ratio"] = (
            agg[f"decay{hl}d_w_debit"] / (agg[f"decay{hl}d_w_credit"] + 1e-9)
        ).astype(np.float32)
        for col in agg.select_dtypes(include=["float64"]).columns:
            agg[col] = agg[col].astype(np.float32)
        rows.append(agg)
        txn.drop(columns=["_w", "_wamt", "_wdebit", "_wcredit"], inplace=True)
        del agg
        gc.collect()

    result = rows[0]
    for f in rows[1:]:
        result = result.merge(f, on=CUSTOMER_ID_COL, how="outer")

    # Cross-half-life ratios reveal trend direction without labels
    if "decay30d_w_count" in result.columns and "decay90d_w_count" in result.columns:
        result["decay_trend_30_90"] = (
            result["decay30d_w_count"] / (result["decay90d_w_count"] + 1e-9)
        ).astype(np.float32)

    if "decay30d_w_count" in result.columns and "decay180d_w_count" in result.columns:
        result["decay_trend_30_180"] = (
            result["decay30d_w_count"] / (result["decay180d_w_count"] + 1e-9)
        ).astype(np.float32)

    txn.drop(columns=["days_ago"], inplace=True)
    result.fillna(0, inplace=True)

    log.info(f"  Decay-weighted features: {result.shape}")
    return result


# ===========================================================================
# 6. [NEW] Full debit/credit decomposition  (per-customer, all-history)
# ===========================================================================

def build_debit_credit_features(txn):
    """
    Full debit and credit decomposition: count, sum, mean, std, min, max
    and ratios between them.  Uses raw transaction rows for accuracy.
    """
    log.info("Building debit/credit decomposition features ...")

    if "debit_credit" not in txn.columns:
        log.warning("  debit_credit column missing — skipping")
        return pd.DataFrame({CUSTOMER_ID_COL: txn[CUSTOMER_ID_COL].unique()})

    debits  = txn[txn["debit_credit"] == "debit"]
    credits = txn[txn["debit_credit"] == "credit"]

    def _agg(df, prefix):
        a = (
            df.groupby(CUSTOMER_ID_COL)[AMOUNT_ABS_COL]
            .agg(
                **{
                    f"{prefix}_count":  "count",
                    f"{prefix}_sum":    "sum",
                    f"{prefix}_mean":   "mean",
                    f"{prefix}_std":    "std",
                    f"{prefix}_median": "median",
                    f"{prefix}_min":    "min",
                    f"{prefix}_max":    "max",
                    f"{prefix}_q25":    lambda x: x.quantile(0.25),
                    f"{prefix}_q75":    lambda x: x.quantile(0.75),
                }
            )
            .reset_index()
        )
        a[f"{prefix}_cv"] = (
            a[f"{prefix}_std"] / (a[f"{prefix}_mean"] + 1e-9)
        ).astype(np.float32)
        for col in a.select_dtypes(include=["float64"]).columns:
            a[col] = a[col].astype(np.float32)
        return a

    d_agg  = _agg(debits,  "dc_debit")
    c_agg  = _agg(credits, "dc_credit")
    result = d_agg.merge(c_agg, on=CUSTOMER_ID_COL, how="outer").fillna(0)

    result["dc_freq_ratio"]   = (
        result["dc_debit_count"] / (result["dc_credit_count"] + 1e-9)
    ).astype(np.float32)

    result["dc_amount_ratio"] = (
        result["dc_debit_sum"] / (result["dc_credit_sum"] + 1e-9)
    ).astype(np.float32)

    result["dc_net_flow"]     = (
        result["dc_credit_sum"] - result["dc_debit_sum"]
    ).astype(np.float32)

    result["dc_debit_dominance"] = (
        result["dc_debit_count"]
        / (result["dc_debit_count"] + result["dc_credit_count"] + 1e-9)
    ).astype(np.float32)

    del debits, credits, d_agg, c_agg
    gc.collect()

    log.info(f"  Debit/credit features: {result.shape}")
    return result


# ===========================================================================
# 7. [NEW] Balance trajectory features
#    Requires a balance column in txn — graceful skip if absent
# ===========================================================================

BALANCE_COL = "StatementBalance"   # adjust if your parquet uses a different name


def build_balance_features(txn):
    """
    Balance trajectory: avg, min, max, EOM balance, volatility,
    overdraft frequency, and recovery speed from overdraft.
    """
    log.info("Building balance trajectory features ...")

    # Try common balance column names
    bal_candidates = ["StatementBalance", "Balance", "RunningBalance",
                      "ClosingBalance", "AccountBalance"]
    bal_col = next((c for c in bal_candidates if c in txn.columns), None)

    if bal_col is None:
        log.warning("  No balance column found — skipping balance features")
        return pd.DataFrame({CUSTOMER_ID_COL: txn[CUSTOMER_ID_COL].unique()})

    txn = txn[[CUSTOMER_ID_COL, DATE_COL, bal_col, "txn_is_month_end"]].copy()
    txn[bal_col] = pd.to_numeric(txn[bal_col], errors="coerce")

    agg = (
        txn.groupby(CUSTOMER_ID_COL)[bal_col]
        .agg(
            bal_mean="mean",
            bal_std="std",
            bal_min="min",
            bal_max="max",
            bal_median="median",
        )
        .reset_index()
    )

    agg["bal_volatility"] = (
        agg["bal_std"] / (agg["bal_mean"].abs() + 1e-9)
    ).astype(np.float32)

    # Overdraft frequency (balance < 0)
    overdraft = (
        txn.groupby(CUSTOMER_ID_COL)[bal_col]
        .apply(lambda x: (x < 0).mean())
        .reset_index(name="bal_overdraft_freq")
    )
    agg = agg.merge(overdraft, on=CUSTOMER_ID_COL, how="left")

    # End-of-month balance (last balance in each month)
    eom = txn[txn["txn_is_month_end"] == 1]
    if len(eom) > 0:
        eom_agg = (
            eom.groupby(CUSTOMER_ID_COL)[bal_col]
            .agg(bal_eom_mean="mean", bal_eom_std="std", bal_eom_min="min")
            .reset_index()
        )
        agg = agg.merge(eom_agg, on=CUSTOMER_ID_COL, how="left")

    # Balance range (breathing room)
    agg["bal_range"] = (agg["bal_max"] - agg["bal_min"]).astype(np.float32)

    for col in agg.select_dtypes(include=["float64"]).columns:
        agg[col] = agg[col].astype(np.float32)

    agg.fillna(0, inplace=True)
    log.info(f"  Balance features: {agg.shape}")
    return agg


# ===========================================================================
# 8. [NEW] Transaction-type behavioral embeddings  (TF-IDF + SVD/clustering)
# ===========================================================================

N_TYPE_COMPONENTS = 10   # SVD components from type freq vectors


def build_type_embedding_features(txn):
    """
    Build per-customer behavioural vectors from transaction categories/types
    using TF-IDF-style normalization, then compress with TruncatedSVD.
    Captures spending pattern diversity and lifestyle segment.
    """
    log.info("Building transaction-type embedding features ...")
    result = pd.DataFrame({CUSTOMER_ID_COL: txn[CUSTOMER_ID_COL].unique()})

    for col, prefix in [
        (CATEGORY_COL, "type_emb"),
        (CHANNEL_COL,  "chan_emb"),
    ]:
        if not col or col not in txn.columns:
            continue

        # Raw frequency pivot
        pivot = (
            txn.groupby([CUSTOMER_ID_COL, col], observed=True)
            .size()
            .unstack(fill_value=0)
            .astype(np.float32)
        )

        # TF (term freq): normalize per customer
        tf = pivot.div(pivot.sum(axis=1) + 1e-9, axis=0)

        # IDF: log(N / df)  — downweight extremely common types
        N_cust  = len(pivot)
        df_freq = (pivot > 0).sum(axis=0)
        idf     = np.log(N_cust / (df_freq + 1)).values.astype(np.float32)
        tfidf   = (tf.values * idf).astype(np.float32)

        # SVD compression
        n_comp = min(N_TYPE_COMPONENTS, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_comp < 1:
            continue
        svd  = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
        emb  = svd.fit_transform(tfidf).astype(np.float32)

        emb_df = pd.DataFrame(
            emb,
            columns=[f"{prefix}_comp_{i}" for i in range(n_comp)],
            index=pivot.index,
        ).reset_index()

        result = result.merge(emb_df, on=CUSTOMER_ID_COL, how="left")

        # Also keep simple fraction features (v2 baseline)
        frac = tf.copy()
        frac.columns = [f"cat_frac_{c}" if prefix == "type_emb" else f"ch_frac_{c}"
                        for c in frac.columns]
        frac = frac.reset_index()
        result = result.merge(frac, on=CUSTOMER_ID_COL, how="left")

        del pivot, tf, tfidf, emb, emb_df, frac
        gc.collect()

    # Raw diversity counts (v2 baseline)
    for col, new_col in [(CHANNEL_COL, "n_channels"), (CATEGORY_COL, "n_categories")]:
        if col and col in txn.columns:
            n_distinct = (
                txn.groupby(CUSTOMER_ID_COL, observed=True)[col]
                .nunique()
                .reset_index(name=new_col)
            )
            result = result.merge(n_distinct, on=CUSTOMER_ID_COL, how="left")

    result.fillna(0, inplace=True)
    for col in result.select_dtypes(include=["float64"]).columns:
        result[col] = result[col].astype(np.float32)

    log.info(f"  Type embedding features: {result.shape}")
    return result


# ===========================================================================
# 9. Recency features  (unchanged from v2)
# ===========================================================================

def build_recency_features(txn):
    log.info("Building recency features ...")

    last_txn = (
        txn.groupby(CUSTOMER_ID_COL)[DATE_COL]
        .max()
        .reset_index(name="last_txn_date")
    )
    last_txn["days_since_last_txn"] = (
        CUTOFF - last_txn["last_txn_date"]
    ).dt.days.astype(np.int16)

    first_txn = (
        txn.groupby(CUSTOMER_ID_COL)[DATE_COL]
        .min()
        .reset_index(name="first_txn_date")
    )
    first_txn["customer_tenure_days"] = (
        CUTOFF - first_txn["first_txn_date"]
    ).dt.days.astype(np.int16)

    result = last_txn[[CUSTOMER_ID_COL, "days_since_last_txn"]].merge(
        first_txn[[CUSTOMER_ID_COL, "customer_tenure_days"]],
        on=CUSTOMER_ID_COL, how="outer"
    )

    txn_sorted = txn.sort_values([CUSTOMER_ID_COL, DATE_COL])
    txn_sorted["prev_date"] = txn_sorted.groupby(CUSTOMER_ID_COL)[DATE_COL].shift(1)
    txn_sorted["days_between"] = (
        txn_sorted[DATE_COL] - txn_sorted["prev_date"]
    ).dt.days

    avg_gap = (
        txn_sorted.groupby(CUSTOMER_ID_COL)["days_between"]
        .mean()
        .reset_index(name="avg_days_between_txn")
    )
    avg_gap["avg_days_between_txn"] = avg_gap["avg_days_between_txn"].astype(np.float32)
    result = result.merge(avg_gap, on=CUSTOMER_ID_COL, how="left")

    del last_txn, first_txn, txn_sorted, avg_gap
    gc.collect()

    log.info(f"  Recency features: {result.shape}")
    return result


# ===========================================================================
# 10. Amount distribution features  (unchanged from v2)
# ===========================================================================

def build_amount_features(txn):
    log.info("Building amount distribution features ...")

    agg = (
        txn.groupby(CUSTOMER_ID_COL)[AMOUNT_ABS_COL]
        .agg(
            amt_total="sum",
            amt_mean="mean",
            amt_median="median",
            amt_std="std",
            amt_max="max",
            amt_min="min",
            amt_q25=lambda x: x.quantile(0.25),
            amt_q75=lambda x: x.quantile(0.75),
            amt_q90=lambda x: x.quantile(0.90),
            amt_skew="skew",
            amt_count="count",
        )
        .reset_index()
    )
    agg["amt_iqr"]        = (agg["amt_q75"] - agg["amt_q25"]).astype(np.float32)
    agg["amt_cv"]         = (agg["amt_std"] / (agg["amt_mean"] + 1e-9)).astype(np.float32)
    agg["amt_large_frac"] = (
        txn.groupby(CUSTOMER_ID_COL)[AMOUNT_ABS_COL]
        .apply(lambda x: (x > x.quantile(0.90)).mean())
        .reset_index(name="tmp")["tmp"]
        .values
    )

    for col in agg.select_dtypes(include=["float64"]).columns:
        agg[col] = agg[col].astype(np.float32)

    log.info(f"  Amount features: {agg.shape}")
    return agg


# ===========================================================================
# 11. Channel / category features  — replaced by build_type_embedding_features
#     Kept for backward compatibility (returns empty to avoid duplication)
# ===========================================================================

def build_channel_category_features(txn):
    """
    Superseded by build_type_embedding_features which adds SVD embeddings
    on top of the same fraction features.  Returns empty frame so callers
    don't need to change.  Actual features are built in assemble_nedbank_data.
    """
    log.info("Channel/category features: delegated to type-embedding builder")
    return pd.DataFrame({CUSTOMER_ID_COL: txn[CUSTOMER_ID_COL].unique()})


# ===========================================================================
# 12. Anomaly features  (unchanged from v2)
# ===========================================================================

def build_anomaly_features(customer_features):
    log.info("Building anomaly features ...")
    feat     = customer_features.copy()
    num_cols = _numeric_cols(feat)

    z_cols = [c for c in num_cols if any(k in c.lower()
              for k in ("txn", "amount", "amt", "vel_", "decay"))]
    if z_cols:
        z_data   = feat[z_cols].fillna(0).values
        z_matrix = np.abs(stats.zscore(z_data, nan_policy="omit"))
        feat["anomaly_zscore_max"]  = z_matrix.max(axis=1).astype(np.float32)
        feat["anomaly_zscore_mean"] = z_matrix.mean(axis=1).astype(np.float32)
        feat["anomaly_zscore_flag"] = (feat["anomaly_zscore_max"] > 3).astype(np.int8)
        del z_data, z_matrix

    iso_data = feat[num_cols].fillna(0)
    iso = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=ISO_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(iso_data)
    feat["anomaly_iso_flag"] = iso.predict(iso_data).astype(np.int8)
    feat["anomaly_iso_raw"]  = (-iso.decision_function(iso_data)).astype(np.float32)
    del iso_data
    gc.collect()

    log.info(f"  Anomaly features: {feat.shape}")
    return feat


# ===========================================================================
# 13. Segmentation features  (unchanged from v2)
# ===========================================================================

def build_segment_features(customer_features):
    log.info(f"Building segment features (k={N_CLUSTERS}, pca={N_PCA_COMPONENTS}) ...")
    feat     = customer_features.copy()
    num_cols = _numeric_cols(feat)

    X = feat[num_cols].fillna(0).values.astype(np.float32)
    X = RobustScaler().fit_transform(X)
    gc.collect()

    n_comp = min(N_PCA_COMPONENTS, X.shape[1], X.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    X_pca  = pca.fit_transform(X).astype(np.float32)
    del X
    gc.collect()

    pca_cols       = [f"pca_comp_{i+1}" for i in range(n_comp)]
    feat[pca_cols] = X_pca

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    km.fit(X_pca)
    feat["customer_segment"]      = km.labels_.astype(np.int8)
    feat["customer_segment_dist"] = km.transform(X_pca).min(axis=1).astype(np.float32)

    seg_dummies = pd.get_dummies(
        feat["customer_segment"].astype(str), prefix="seg", drop_first=False
    )
    feat = pd.concat([feat, seg_dummies], axis=1)

    del X_pca, km
    gc.collect()

    log.info(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return feat


# ===========================================================================
# 14. Main assembly function
# ===========================================================================

def assemble_nedbank_data(txn_clean, fin_clean, demo_clean,
                          train_labels, test_ids, use_prophet=False):
    log.info("Assembling Nedbank_data (v3) ...")

    # Step 1: temporal features
    txn_clean = build_temporal_features(txn_clean)
    gc.collect()

    # Step 2: customer aggregates (holiday, streaks, debit/credit monthly)
    cust_agg = build_customer_aggregates(txn_clean.copy())
    gc.collect()

    # Step 3: type/channel embeddings (replaces old channel_category_features)
    ch_cat = build_type_embedding_features(txn_clean)
    gc.collect()

    # Step 4: recency features
    recency = build_recency_features(txn_clean)
    gc.collect()

    # Step 5: amount distribution features
    amt_feats = build_amount_features(txn_clean)
    gc.collect()

    # Step 6: [NEW] velocity features
    velocity = build_velocity_features(txn_clean)
    gc.collect()

    # Step 7: [NEW] decay-weighted features
    decay_feats = build_decay_weighted_features(txn_clean)
    gc.collect()

    # Step 8: [NEW] full debit/credit decomposition
    dc_feats = build_debit_credit_features(txn_clean)
    gc.collect()

    # Step 9: [NEW] balance trajectory
    bal_feats = build_balance_features(txn_clean)
    del txn_clean
    gc.collect()

    # Step 10: latest financial snapshot per customer
    date_cols = [c for c in fin_clean.columns if "date" in c.lower() or "Date" in c]
    if date_cols:
        fin_clean.sort_values([CUSTOMER_ID_COL, date_cols[0]], inplace=True)
    fin_latest = fin_clean.groupby(CUSTOMER_ID_COL).last().reset_index()
    del fin_clean
    gc.collect()

    # Step 11: merge all tables
    Nedbank_data = (
        cust_agg
        .merge(ch_cat,      on=CUSTOMER_ID_COL, how="left")
        .merge(recency,     on=CUSTOMER_ID_COL, how="left")
        .merge(amt_feats,   on=CUSTOMER_ID_COL, how="left")
        .merge(velocity,    on=CUSTOMER_ID_COL, how="left")
        .merge(decay_feats, on=CUSTOMER_ID_COL, how="left")
        .merge(dc_feats,    on=CUSTOMER_ID_COL, how="left")
        .merge(bal_feats,   on=CUSTOMER_ID_COL, how="left")
        .merge(fin_latest,  on=CUSTOMER_ID_COL, how="left")
        .merge(demo_clean,  on=CUSTOMER_ID_COL, how="left")
    )
    del cust_agg, ch_cat, recency, amt_feats, velocity, decay_feats, dc_feats, bal_feats, fin_latest, demo_clean
    gc.collect()

    # Step 12: anomaly features
    Nedbank_data = build_anomaly_features(Nedbank_data)
    gc.collect()

    # Step 13: segmentation features
    Nedbank_data = build_segment_features(Nedbank_data)
    gc.collect()

    # Step 14: attach labels and split flag
    all_ids = pd.concat([
        train_labels[[CUSTOMER_ID_COL, "next_3m_txn_count"]].assign(split="train"),
        test_ids[[CUSTOMER_ID_COL]].assign(split="test"),
    ], ignore_index=True)
    Nedbank_data = all_ids.merge(Nedbank_data, on=CUSTOMER_ID_COL, how="left")

    log.info(f"Nedbank_data assembled: {Nedbank_data.shape}")
    log.info(f"  Train rows: {(Nedbank_data['split'] == 'train').sum()}")
    log.info(f"  Test  rows: {(Nedbank_data['split'] == 'test').sum()}")
    return Nedbank_data
