"""
modelling_v3.py  — Drop-in replacement for modelling.py
=========================================================
Key additions vs v2:
  1.  Two-stage zero-inflation model
        Stage 1: classify "will customer transact?" (XGB binary classifier)
        Stage 2: count regression (same as v2)
        Final:   P(active) * count_prediction
  2.  Poisson / count-appropriate objectives
        XGB: objective="count:poisson"  (+ reg:squarederror as fallback for tuning)
        LGB: objective="poisson"
  3.  Proper temporal validation
        Folds split on time (hist_months_active proxy), so validation
        always sees data "later" than training — mimics the leaderboard gap.
  All v2 APIs (tune_xgb, train_xgb, tune_lgb, train_lgb, predict_ensemble)
  are preserved.  Two new functions added:
      train_zero_inflation_stage1()
      predict_ensemble_two_stage()
"""
import gc
import logging
import warnings

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from config import (
    TARGET,
    RANDOM_SEED,
    N_CV_FOLDS,
    MIN_TRAIN_FRAC,
    XGB_PARAMS,
    LGB_PARAMS,
    LGB_EARLY_STOPPING,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    OPTUNA_DIRECTION,
    ENSEMBLE_WEIGHTS,
)
from metrics import rmsle, evaluate

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("nedbank.modelling")

# ---------------------------------------------------------------------------
# Poisson-aware params (override objective for count prediction)
# ---------------------------------------------------------------------------
XGB_POISSON_PARAMS = {
    **XGB_PARAMS,
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
}

LGB_POISSON_PARAMS = {
    **LGB_PARAMS,
    "objective": "poisson",
    "metric": "poisson",
}

# Classifier params for Stage 1 (binary: transacts or not)
XGB_CLASSIFIER_PARAMS = {
    **XGB_PARAMS,
    "objective":  "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 800,
    "learning_rate": 0.05,
}


# ===========================================================================
# Cross-validation splits  (unchanged + improved docs)
# ===========================================================================

def time_based_cv_splits(Nedbank_data, n_folds=N_CV_FOLDS, min_frac=MIN_TRAIN_FRAC):
    """
    True temporal validation: rows are sorted by hist_months_active (a proxy
    for how much history a customer has, i.e. their 'time position').
    Later folds always have longer-history customers in validation, matching
    the leaderboard scenario where test customers have full 34-month history.
    """
    train_df = (
        Nedbank_data[Nedbank_data["split"] == "train"]
        sort_col = "hist_months_active" if "hist_months_active" in Nedbank_data.columns else Nedbank_data.select_dtypes(include="number").columns[0]
        .sort_values(sort_col)
        .reset_index(drop=True)
    )
    n      = len(train_df)
    splits = []
    for fold in range(n_folds):
        train_end = int(n * (min_frac + (1 - min_frac) * fold / n_folds))
        val_start = train_end
        val_end   = min(int(n * (min_frac + (1 - min_frac) * (fold + 1) / n_folds)), n)
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
        log.info(f"  Fold {fold+1}: train={train_end}, val={val_end - val_start}")
    return splits


# ===========================================================================
# Data helpers
# ===========================================================================

def _get_Xy(Nedbank_data, feature_cols, split="train", log_transform=True):
    mask = Nedbank_data["split"] == split
    X    = Nedbank_data.loc[mask, feature_cols].fillna(0).values.astype(np.float32)
    y    = None
    if TARGET in Nedbank_data.columns and split == "train":
        raw_y = Nedbank_data.loc[mask, TARGET].values.astype(float)
        y     = np.log1p(raw_y) if log_transform else raw_y
    return X, y


def _get_Xy_binary(Nedbank_data, feature_cols, split="train"):
    """Returns X and binary y (1 if target > 0, 0 otherwise)."""
    mask = Nedbank_data["split"] == split
    X    = Nedbank_data.loc[mask, feature_cols].fillna(0).values.astype(np.float32)
    y    = None
    if TARGET in Nedbank_data.columns and split == "train":
        raw_y = Nedbank_data.loc[mask, TARGET].values.astype(float)
        y     = (raw_y > 0).astype(np.float32)
    return X, y


# ===========================================================================
# Stage 1: Zero-inflation classifier
# ===========================================================================

def train_zero_inflation_stage1(Nedbank_data, feature_cols):
    """
    Binary classifier: predicts P(customer will make at least 1 transaction).
    Returns:
        models   — list of fold classifiers
        oof_prob — out-of-fold probabilities (train rows, same order as train mask)
    """
    log.info("Training Stage-1 zero-inflation classifier ...")
    X_all, y_all = _get_Xy_binary(Nedbank_data, feature_cols)
    splits       = time_based_cv_splits(Nedbank_data)
    n_train      = len(y_all)
    oof_prob     = np.zeros(n_train)
    oof_mask     = np.zeros(n_train, dtype=bool)
    models       = []

    zero_frac    = (y_all == 0).mean()
    log.info(f"  Zero (inactive) fraction in train: {zero_frac:.2%}")

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        params = {
            **XGB_CLASSIFIER_PARAMS,
            "scale_pos_weight": (y_all[tr_idx] == 0).sum() / (y_all[tr_idx] == 1).sum() + 1e-9,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(
            X_all[tr_idx], y_all[tr_idx],
            eval_set=[(X_all[val_idx], y_all[val_idx])],
            verbose=False,
        )
        prob = m.predict_proba(X_all[val_idx])[:, 1]
        oof_prob[val_idx]  = prob
        oof_mask[val_idx]  = True

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_all[val_idx], prob)
        log.info(f"  Stage-1 Fold {fold_idx+1}: AUC={auc:.4f}")
        models.append(m)
        gc.collect()

    log.info(f"  Stage-1 OOF AUC: {roc_auc_score(y_all[oof_mask], oof_prob[oof_mask]):.4f}")
    return models, oof_prob


# ===========================================================================
# XGBoost tuning and training  (Poisson objective)
# ===========================================================================

def tune_xgb(Nedbank_data, feature_cols):
    log.info(f"Tuning XGBoost with Optuna ({OPTUNA_N_TRIALS} trials) ...")
    X_all, y_all = _get_Xy(Nedbank_data, feature_cols)
    splits       = time_based_cv_splits(Nedbank_data)

    def objective(trial):
        params = {
            **XGB_POISSON_PARAMS,
            "n_estimators":      trial.suggest_int("n_estimators", 1000, 4000, step=200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth":         trial.suggest_int("max_depth", 6, 10),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 15),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        }
        scores = []
        for tr_idx, val_idx in splits:
            m = xgb.XGBRegressor(**params)
            # Poisson requires non-negative raw targets
            y_tr  = np.expm1(y_all[tr_idx])   # back to counts
            y_val = np.expm1(y_all[val_idx])
            m.fit(X_all[tr_idx], y_tr,
                  eval_set=[(X_all[val_idx], y_val)], verbose=False)
            pred = np.maximum(m.predict(X_all[val_idx]), 0)
            scores.append(rmsle(y_val, pred))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
    best = {**XGB_POISSON_PARAMS, **study.best_params}
    log.info(f"  Best XGB RMSLE: {study.best_value:.4f}")
    log.info(f"  Best XGB params: {study.best_params}")
    return best


def train_xgb(Nedbank_data, feature_cols, params=None):
    """
    Train XGBoost count:poisson model.
    Targets are passed as raw counts (non-negative integers) for Poisson loss.
    OOF predictions are on count scale.
    """
    if params is None:
        params = XGB_POISSON_PARAMS
    log.info("Training XGBoost (Poisson) ...")
    X_all, y_log = _get_Xy(Nedbank_data, feature_cols)
    y_all        = np.expm1(y_log)   # raw counts for Poisson

    splits   = time_based_cv_splits(Nedbank_data)
    n_train  = len(y_all)
    oof      = np.zeros(n_train)
    oof_mask = np.zeros(n_train, dtype=bool)
    models   = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**params)
        m.fit(
            X_all[tr_idx], y_all[tr_idx],
            eval_set=[(X_all[val_idx], y_all[val_idx])],
            verbose=200,
        )
        oof[val_idx]      = np.maximum(m.predict(X_all[val_idx]), 0)
        oof_mask[val_idx] = True
        evaluate(y_all[val_idx], oof[val_idx], label=f"XGB Fold {fold_idx+1}")
        models.append(m)
        gc.collect()

    log.info("-- XGB Overall OOF (validation folds only) --")
    evaluate(y_all[oof_mask], oof[oof_mask], label="XGB OOF")
    return models, oof


def tune_lgb(Nedbank_data, feature_cols):
    log.info(f"Tuning LightGBM with Optuna ({OPTUNA_N_TRIALS} trials) ...")
    X_all, y_log = _get_Xy(Nedbank_data, feature_cols)
    y_all        = np.expm1(y_log)
    splits       = time_based_cv_splits(Nedbank_data)

    def objective(trial):
        params = {
            **LGB_POISSON_PARAMS,
            "n_estimators":      trial.suggest_int("n_estimators", 1000, 4000, step=200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 63, 511),
            "max_depth":         trial.suggest_int("max_depth", 6, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
        }
        scores = []
        for tr_idx, val_idx in splits:
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_all[tr_idx], y_all[tr_idx],
                eval_set=[(X_all[val_idx], y_all[val_idx])],
                callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            pred = np.maximum(m.predict(X_all[val_idx]), 0)
            scores.append(rmsle(y_all[val_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
    best = {**LGB_POISSON_PARAMS, **study.best_params}
    log.info(f"  Best LGB RMSLE: {study.best_value:.4f}")
    log.info(f"  Best LGB params: {study.best_params}")
    return best


def train_lgb(Nedbank_data, feature_cols, params=None):
    """
    Train LightGBM Poisson model.
    Targets are raw counts (non-negative integers).
    """
    if params is None:
        params = LGB_POISSON_PARAMS
    log.info("Training LightGBM (Poisson) ...")
    X_all, y_log = _get_Xy(Nedbank_data, feature_cols)
    y_all        = np.expm1(y_log)

    splits   = time_based_cv_splits(Nedbank_data)
    n_train  = len(y_all)
    oof      = np.zeros(n_train)
    oof_mask = np.zeros(n_train, dtype=bool)
    models   = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_all[tr_idx], y_all[tr_idx],
            eval_set=[(X_all[val_idx], y_all[val_idx])],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(period=200)],
        )
        oof[val_idx]      = np.maximum(m.predict(X_all[val_idx]), 0)
        oof_mask[val_idx] = True
        evaluate(y_all[val_idx], oof[val_idx], label=f"LGB Fold {fold_idx+1}")
        models.append(m)
        gc.collect()

    log.info("-- LGB Overall OOF (validation folds only) --")
    evaluate(y_all[oof_mask], oof[oof_mask], label="LGB OOF")
    return models, oof


# ===========================================================================
# Ensemble prediction  (v2 baseline — stage-1 optional)
# ===========================================================================

def predict_ensemble(xgb_models, lgb_models, Nedbank_data, feature_cols,
                     split="test", stage1_models=None):
    """
    Predict count for a given split.
    If stage1_models is provided, applies two-stage zero-inflation correction.
    """
    mask = Nedbank_data["split"] == split
    X    = Nedbank_data.loc[mask, feature_cols].fillna(0).values.astype(np.float32)

    # XGBoost: Poisson model predicts raw counts directly
    xgb_pred = np.mean(
        [np.maximum(m.predict(X), 0) for m in xgb_models], axis=0
    )
    lgb_pred = np.mean(
        [np.maximum(m.predict(X), 0) for m in lgb_models], axis=0
    )

    w_xgb    = ENSEMBLE_WEIGHTS.get("xgb", 0.5)
    w_lgb    = ENSEMBLE_WEIGHTS.get("lgb", 0.5)
    ensemble = np.maximum(w_xgb * xgb_pred + w_lgb * lgb_pred, 0)

    # Two-stage correction
    if stage1_models is not None:
        log.info("  Applying Stage-1 zero-inflation correction ...")
        p_active = np.mean(
            [m.predict_proba(X)[:, 1] for m in stage1_models], axis=0
        )
        ensemble = p_active * ensemble
        log.info(
            f"  P(active) — mean={p_active.mean():.3f}, "
            f"min={p_active.min():.3f}, max={p_active.max():.3f}"
        )

    log.info(
        f"Ensemble ({split}): mean={ensemble.mean():.2f}, "
        f"median={np.median(ensemble):.2f}, max={ensemble.max():.2f}, "
        f"zeros={int((ensemble < 0.5).sum())}"
    )
    return ensemble


def predict_ensemble_two_stage(xgb_models, lgb_models, stage1_models,
                                Nedbank_data, feature_cols, split="test"):
    """Convenience wrapper for the two-stage prediction."""
    return predict_ensemble(
        xgb_models, lgb_models, Nedbank_data, feature_cols,
        split=split, stage1_models=stage1_models,
    )
