"""
feature_selection_v5.py
========================
Lightweight pre-filter run once before modelling.
Heavy per-fold selection is handled inside modelling_v5.run_pipeline().

Changes vs v4:
  - Removed model-based (XGB) selection stage (heavy, redundant with per-fold selection)
  - Removed multi-stage selection unless proven useful
  - Ranking: corr(feature, log1p(target)) — fast and RMSLE-consistent
  - Correlation threshold relaxed: only near-identical cols dropped (0.97)
  - Frequency encoding reminder added (LabelEncoder warning)
"""

import gc
import logging

import numpy as np
import pandas as pd

from config import CUSTOMER_ID_COL, TARGET, RANDOM_SEED

log = logging.getLogger("nedbank.feature_selection_v5")


def _get_numeric_feature_cols(Nedbank_data: pd.DataFrame) -> list:
    exclude = {CUSTOMER_ID_COL, "split", TARGET, "customer_segment"}
    valid_dtypes = [np.float64, np.float32, np.int64, np.int32,
                    np.int16, np.int8, np.uint8, bool]
    return [
        c for c in Nedbank_data.columns
        if c not in exclude and Nedbank_data[c].dtype in valid_dtypes
    ]


def select_features(
    Nedbank_data:          pd.DataFrame,
    variance_threshold:    float = 1e-6,
    correlation_threshold: float = 0.97,
    max_features:          int   = 100,
) -> tuple:
    """
    Light pre-filter:
      Stage 1 — Remove zero-variance
      Stage 2 — Remove near-identical (corr > 0.97); keep higher |corr(log1p(y))|
      Stage 3 — Rank by |corr(feature, log1p(y))|; keep top max_features

    Returns (Nedbank_data_filtered, selected_feature_cols).
    """
    log.info("Running lightweight feature selection ...")

    always_keep = {CUSTOMER_ID_COL, "split", TARGET, "customer_segment"}
    raw_cols = _get_numeric_feature_cols(Nedbank_data)
    log.info(f"  Starting features: {len(raw_cols)}")

    train_mask = Nedbank_data["split"] == "train"
    X_train    = Nedbank_data.loc[train_mask, raw_cols].fillna(0)
    y_train    = Nedbank_data.loc[train_mask, TARGET].values.astype(float)
    log1p_y    = np.log1p(y_train)

    # Stage 1: variance
    var         = X_train.var()
    after_var   = var[var > variance_threshold].index.tolist()
    log.info(f"  After variance filter: {len(after_var)} "
             f"(dropped {len(raw_cols) - len(after_var)})")

    # Stage 2: correlation (sampled for speed)
    n_sample = min(5000, len(X_train))
    sample   = X_train[after_var].sample(n_sample, random_state=RANDOM_SEED)
    corr_mat = sample.corr().abs()

    target_corr = X_train[after_var].corrwith(
        pd.Series(log1p_y, index=X_train.index)
    ).abs()

    upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    drop_set: set = set()
    for col in upper.columns:
        if col in drop_set:
            continue
        partners = upper.index[upper[col] > correlation_threshold].tolist()
        for partner in partners:
            if partner in drop_set:
                continue
            if target_corr.get(col, 0) >= target_corr.get(partner, 0):
                drop_set.add(partner)
            else:
                drop_set.add(col)

    after_corr = [c for c in after_var if c not in drop_set]
    log.info(f"  After correlation filter: {len(after_corr)} "
             f"(dropped {len(drop_set)})")
    del corr_mat, upper, sample
    gc.collect()

    # Stage 3: rank by |corr(log1p(y))|
    if len(after_corr) > max_features:
        ranked = (
            target_corr[after_corr]
            .sort_values(ascending=False)
            .head(max_features)
        )
        selected = ranked.index.tolist()
    else:
        selected = after_corr

    log.info(f"  Final selection: {len(selected)} features")

    # Top-20 by correlation (informational)
    top20 = target_corr[selected].sort_values(ascending=False).head(20)
    log.info(f"\n  Top 20 by |corr(log1p(y))|:\n{top20.to_string()}")

    keep_cols = list(always_keep) + selected
    keep_cols = [c for c in keep_cols if c in Nedbank_data.columns]
    Nedbank_data_sel = Nedbank_data[keep_cols].copy()
    gc.collect()

    log.info(f"  Final shape: {Nedbank_data_sel.shape}")
    return Nedbank_data_sel, selected
