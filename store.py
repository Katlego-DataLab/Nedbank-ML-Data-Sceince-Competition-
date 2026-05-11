import gc
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TARGET, CUSTOMER_ID_COL

log = logging.getLogger("nedbank.store")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUTPUT_DIR / "nedbank_feature_store.sqlite"


def export_to_sqlite(Nedbank_data, db_path=DB_PATH):
    log.info(f"Exporting feature store to SQLite: {db_path} ...")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        Nedbank_data.to_sql("nedbank_features", con, if_exists="replace", index=False)

        all_cols  = Nedbank_data.columns.tolist()
        id_target = [CUSTOMER_ID_COL, "split", TARGET]

        agg_cols  = id_target + [c for c in all_cols
                     if any(c.startswith(p) for p in
                            ["hist_", "roll", "lag", "mom_", "holiday_"])]
        anm_cols  = id_target + [c for c in all_cols if "anomaly" in c]
        seg_cols  = id_target + [c for c in all_cols
                     if c.startswith(("pca_", "seg_", "customer_seg"))]
        prph_cols = id_target + [c for c in all_cols if c.startswith("prophet_")]

        def _cols_sql(cols):
            unique = list(dict.fromkeys(cols))
            return ", ".join(f'"{c}"' for c in unique if c in all_cols)

        views = {
            "v_train":
                "SELECT * FROM nedbank_features WHERE split = 'train'",
            "v_test":
                "SELECT * FROM nedbank_features WHERE split = 'test'",
            "v_aggregate_features":
                f"SELECT {_cols_sql(agg_cols)} FROM nedbank_features",
            "v_anomaly_features":
                f"SELECT {_cols_sql(anm_cols)} FROM nedbank_features",
            "v_segment_features":
                f"SELECT {_cols_sql(seg_cols)} FROM nedbank_features",
            "v_prophet_features":
                f"SELECT {_cols_sql(prph_cols)} FROM nedbank_features",
            "v_submission_ready": (
                f'SELECT "{CUSTOMER_ID_COL}", 0.0 AS {TARGET} '
                f"FROM nedbank_features WHERE split = 'test'"
            ),
        }

        cur = con.cursor()
        for vname, vsql in views.items():
            cur.execute(f"DROP VIEW IF EXISTS {vname}")
            try:
                cur.execute(f"CREATE VIEW {vname} AS {vsql}")
                log.info(f"  Created view: {vname}")
            except sqlite3.OperationalError as exc:
                log.warning(f"  Could not create view {vname}: {exc}")
        con.commit()

    log.info("SQLite export complete.")


def save_feature_store_parquet(Nedbank_data):
    path = OUTPUT_DIR / "feature_store.parquet"
    Nedbank_data.to_parquet(path, index=False)
    log.info(f"Feature store saved to parquet: {path}")
    return path


def update_predictions(predictions, db_path=DB_PATH):
    required = {CUSTOMER_ID_COL, "predicted_next_3m_txn_count_raw",
                "predicted_next_3m_txn_count_log1p"}
    missing  = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")

    with sqlite3.connect(db_path) as con:
        predictions.to_sql("predictions", con, if_exists="replace", index=False)
        cur = con.cursor()
        cur.execute("DROP VIEW IF EXISTS v_submission_ready")
        cur.execute(
            f"""
            CREATE VIEW v_submission_ready AS
            SELECT "{CUSTOMER_ID_COL}",
                   predicted_next_3m_txn_count_log1p AS {TARGET}
            FROM predictions
            """
        )
        con.commit()

    # Also save predictions as CSV for download
    pred_path = OUTPUT_DIR / "predictions.csv"
    predictions.to_csv(pred_path, index=False)
    log.info(f"Predictions saved: {pred_path}")
    log.info("SQLite v_submission_ready updated.")


def build_submission(Nedbank_data, test_preds, sample_submission,
                     save_path=OUTPUT_DIR / "submission.csv"):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    test_customers = (
        Nedbank_data.loc[Nedbank_data["split"] == "test", CUSTOMER_ID_COL]
        .reset_index(drop=True)
    )

    if len(test_customers) != len(test_preds):
        raise ValueError(
            f"Length mismatch: {len(test_customers)} customers, "
            f"{len(test_preds)} predictions."
        )

    # Apply log1p as required by competition rules
    # Rules: "You must submit np.log1p(y_pred). Do not submit raw counts."
    log1p_preds = np.log1p(np.maximum(test_preds, 0))

    raw_sub = pd.DataFrame({
        CUSTOMER_ID_COL: test_customers,
        TARGET:          log1p_preds,   # floats like 3.456, NOT integers
    })

    sample_id_col = sample_submission.columns[0]
    submission = (
        sample_submission[[sample_id_col]]
        .merge(
            raw_sub.rename(columns={CUSTOMER_ID_COL: sample_id_col}),
            on=sample_id_col,
            how="left",
        )
    )
    submission[TARGET] = submission[TARGET].fillna(0.0)

    # Validate before saving
    assert len(submission) == len(sample_submission), \
        f"Row count mismatch: expected {len(sample_submission)}, got {len(submission)}"
    assert submission.isnull().sum().sum() == 0, \
        "Submission has null values."

    submission.to_csv(save_path, index=False)

    log.info(f"Submission saved: {save_path}  shape={submission.shape}")
    log.info(
        f"  log1p values — mean={log1p_preds.mean():.4f}, "
        f"median={np.median(log1p_preds):.4f}, "
        f"min={log1p_preds.min():.4f}, max={log1p_preds.max():.4f}"
    )
    log.info(
        f"  Raw scale — mean={test_preds.mean():.1f}, "
        f"max={test_preds.max():.1f}, zeros={int((test_preds < 0.5).sum())}"
    )
    return submission
