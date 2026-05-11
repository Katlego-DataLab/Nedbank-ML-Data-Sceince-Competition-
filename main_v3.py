"""
main_v3.py  — Drop-in replacement for main.py
================================================
What changed from v2:
  - Uses feature_engineering_v3.py (velocity, decay, debit/credit, balance,
    type embeddings, holiday granularity, consistency/streak features)
  - Uses modelling_v3.py (Poisson objectives, two-stage zero-inflation,
    temporal validation)
  - Proper OOF evaluation on count scale (no log transform mismatch)
  - Two-stage final prediction (Stage-1 activity classifier × count regressor)

Usage:
    python main_v3.py --no-tune --no-prophet    # fastest, strong baseline
    python main_v3.py --no-prophet              # with Optuna tuning
    python main_v3.py                           # full run with Prophet
"""
import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader               import load_raw_data, clean_transactions, clean_financials, clean_demographics
from feature_engineering_v3    import assemble_nedbank_data   # ← v3 module
from feature_selection_v5      import select_features
from modelling_v3              import (                        # ← v3 module
    tune_xgb, train_xgb,
    tune_lgb, train_lgb,
    train_zero_inflation_stage1,
    predict_ensemble,
    predict_ensemble_two_stage,
    time_based_cv_splits,
)
from metrics    import evaluate as _eval
from store      import (
    export_to_sqlite,
    save_feature_store_parquet,
    update_predictions,
    build_submission,
)
from diagnostics import (
    plot_feature_importance,
    plot_prediction_distribution,
    plot_cv_scores,
    plot_segment_distribution,
    print_cv_summary,
)

warnings.filterwarnings("ignore")


def setup_logging():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.OUTPUT_DIR / "pipeline.log", mode="w"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Nedbank Banking on Behaviour pipeline v3")
    parser.add_argument("--no-tune",        action="store_true", help="Skip Optuna tuning")
    parser.add_argument("--no-prophet",     action="store_true", help="Skip Prophet features")
    parser.add_argument("--no-two-stage",   action="store_true", help="Skip Stage-1 zero-inflation")
    parser.add_argument("--trials", type=int, default=config.OPTUNA_N_TRIALS,
                        help="Optuna trials per model")
    return parser.parse_args()


def main():
    setup_logging()
    log = logging.getLogger("nedbank.main")
    args = parse_args()
    config.OPTUNA_N_TRIALS = args.trials

    log.info("=" * 70)
    log.info("  Nedbank 'Banking on Behaviour' — Pipeline v3")
    log.info(f"  Optuna tuning  : {'OFF' if args.no_tune else 'ON'}")
    log.info(f"  Prophet        : {'OFF' if args.no_prophet else 'ON'}")
    log.info(f"  Two-stage      : {'OFF' if args.no_two_stage else 'ON'}")
    log.info("=" * 70)

    # 1. Load and clean
    raw        = load_raw_data()
    txn_clean  = clean_transactions(raw["transactions_raw"])
    del raw["transactions_raw"]
    gc.collect()

    fin_clean  = clean_financials(raw["financials"])
    del raw["financials"]
    gc.collect()

    demo_clean = clean_demographics(raw["demographics"])
    del raw["demographics"]
    gc.collect()

    # 2. Feature engineering (v3)
    Nedbank_data = assemble_nedbank_data(
        txn_clean    = txn_clean,
        fin_clean    = fin_clean,
        demo_clean   = demo_clean,
        train_labels = raw["train_labels"],
        test_ids     = raw["test_ids"],
        use_prophet  = not args.no_prophet,
    )
    del txn_clean, fin_clean, demo_clean
    gc.collect()

    # 3. Save feature store
    save_feature_store_parquet(Nedbank_data)
    export_to_sqlite(Nedbank_data)

    # 4. Feature selection
    Nedbank_data, feature_cols = select_features(Nedbank_data)
    log.info(f"Selected {len(feature_cols)} features for modelling.")
    gc.collect()

    # 5. Hyperparameter tuning
    if args.no_tune:
        log.info("Tuning skipped — using default Poisson hyperparameters.")
        from modelling_v3 import XGB_POISSON_PARAMS, LGB_POISSON_PARAMS
        xgb_params = XGB_POISSON_PARAMS
        lgb_params = LGB_POISSON_PARAMS
    else:
        xgb_params = tune_xgb(Nedbank_data, feature_cols)
        lgb_params = tune_lgb(Nedbank_data, feature_cols)

    # 6. Stage-1 zero-inflation classifier
    stage1_models = None
    if not args.no_two_stage:
        stage1_models, stage1_oof = train_zero_inflation_stage1(
            Nedbank_data, feature_cols
        )
        gc.collect()

    # 7. Count regression models (Poisson)
    xgb_models, xgb_oof = train_xgb(Nedbank_data, feature_cols, params=xgb_params)
    lgb_models, lgb_oof = train_lgb(Nedbank_data, feature_cols, params=lgb_params)
    gc.collect()

    # 8. CV metrics — on raw count scale
    train_mask = Nedbank_data["split"] == "train"
    train_y    = Nedbank_data.loc[train_mask, config.TARGET].values.astype(float)
    splits     = time_based_cv_splits(Nedbank_data)

    fold_metrics_xgb, fold_metrics_lgb = [], []
    for _, val_idx in splits:
        fold_metrics_xgb.append(_eval(train_y[val_idx], xgb_oof[val_idx]))
        fold_metrics_lgb.append(_eval(train_y[val_idx], lgb_oof[val_idx]))

    print_cv_summary(fold_metrics_xgb, fold_metrics_lgb)

    # Ensemble OOF
    w_xgb        = config.ENSEMBLE_WEIGHTS["xgb"]
    w_lgb        = config.ENSEMBLE_WEIGHTS["lgb"]
    ensemble_oof  = w_xgb * xgb_oof + w_lgb * lgb_oof

    # Apply Stage-1 correction to OOF if available
    if stage1_models is not None:
        ensemble_oof = stage1_oof * ensemble_oof

    n_train  = len(train_y)
    oof_mask = np.zeros(n_train, dtype=bool)
    for _, val_idx in splits:
        oof_mask[val_idx] = True

    log.info("-- Ensemble OOF (validation folds only, count scale) --")
    _eval(train_y[oof_mask], ensemble_oof[oof_mask], label="Ensemble OOF")

    # 9. Diagnostics
    plot_feature_importance(xgb_models, lgb_models, feature_cols)
    plot_prediction_distribution(train_y[oof_mask], ensemble_oof[oof_mask],
                                 label="Ensemble OOF")
    plot_cv_scores(fold_metrics_xgb, fold_metrics_lgb)
    plot_segment_distribution(Nedbank_data)

    # 10. Test predictions (two-stage if Stage-1 available)
    test_preds = predict_ensemble(
        xgb_models, lgb_models, Nedbank_data, feature_cols,
        split="test",
        stage1_models=stage1_models,
    )

    # 11. Save predictions
    test_mask      = Nedbank_data["split"] == "test"
    test_customers = (
        Nedbank_data.loc[test_mask, config.CUSTOMER_ID_COL]
        .reset_index(drop=True)
    )
    update_predictions(pd.DataFrame({
        config.CUSTOMER_ID_COL:               test_customers,
        "predicted_next_3m_txn_count_raw":    test_preds,
        "predicted_next_3m_txn_count_log1p":  np.log1p(np.maximum(test_preds, 0)),
    }))

    # 12. Build and save submission.csv
    submission = build_submission(
        Nedbank_data      = Nedbank_data,
        test_preds        = test_preds,
        sample_submission = raw["sample_submission"],
    )

    # 13. Final summary
    log.info("=" * 70)
    log.info("Pipeline v3 complete. Files ready in outputs/:")
    log.info("  submission.csv           <- upload to Zindi")
    log.info("  feature_store.parquet    <- full feature matrix")
    log.info("  predictions.csv          <- raw + log1p predictions")
    log.info("  feature_importance.csv   <- model diagnostics")
    log.info("  cv_scores.csv            <- cross-validation results")
    log.info("  pipeline.log             <- full run log")
    log.info("=" * 70)

    xgb_mean = np.mean([m["RMSLE"] for m in fold_metrics_xgb])
    lgb_mean = np.mean([m["RMSLE"] for m in fold_metrics_lgb])
    log.info(f"  XGB mean RMSLE: {xgb_mean:.4f}")
    log.info(f"  LGB mean RMSLE: {lgb_mean:.4f}")

    return Nedbank_data, {"xgb": xgb_models, "lgb": lgb_models,
                           "stage1": stage1_models}, submission


if __name__ == "__main__":
    Nedbank_data, models, submission = main()
