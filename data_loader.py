import gc
import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_DIR,
    CUSTOMER_ID_COL,
    DATE_COL,
    AMOUNT_COL,
    AMOUNT_ABS_COL,
    DEBIT_CREDIT_COL,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("nedbank.loader")


def reduce_mem(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    return df


def load_raw_data():
    log.info("Loading raw data files ...")
    raw = {}

    # ── Transactions: memory-safe chunk loading ───────────────────
    log.info("  Loading transactions (memory optimised) ...")

    # Peek at actual columns first
    sample = pd.read_parquet(
        DATA_DIR / "transactions_features.parquet",
        engine="pyarrow",
    ).head(1)
    actual_cols = sample.columns.tolist()
    log.info(f"  Actual parquet columns: {actual_cols}")
    del sample
    gc.collect()

    # Only keep columns that exist in the file
    wanted = [
        CUSTOMER_ID_COL,
        DATE_COL,
        AMOUNT_COL,
        "IsDebitCredit",
        "TransactionBatchDescription",
        "TransactionTypeDescription",
    ]
    keep_cols = [c for c in wanted if c in actual_cols]
    log.info(f"  Keeping columns: {keep_cols}")

    # Load in chunks to avoid RAM crash
    parquet_file = pq.ParquetFile(DATA_DIR / "transactions_features.parquet")
    chunks = []
    for batch in parquet_file.iter_batches(batch_size=100_000, columns=keep_cols):
        chunk = batch.to_pandas()
        chunk = reduce_mem(chunk)
        chunks.append(chunk)
        gc.collect()

    txn = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # Sample if still too large
    log.info(f"  Loaded {len(txn):,} rows")
    if len(txn) > 300_000:
        log.info("  Sampling 30% of customers to fit in RAM ...")
        customers = txn[CUSTOMER_ID_COL].unique()
        rng = np.random.RandomState(42)
        sample_customers = rng.choice(
            customers,
            size=int(len(customers) * 0.3),
            replace=False,
        )
        txn = txn[txn[CUSTOMER_ID_COL].isin(sample_customers)].copy()
        log.info(f"  After sampling: {len(txn):,} rows")
        gc.collect()

    txn = reduce_mem(txn)
    raw["transactions_raw"] = txn
    log.info(f"  transactions_raw: {txn.shape}")
    del txn
    gc.collect()

    # ── Other files ───────────────────────────────────────────────
    raw["financials"]        = pd.read_parquet(DATA_DIR / "financials_features.parquet")
    raw["demographics"]      = pd.read_parquet(DATA_DIR / "demographics_clean.parquet")
    raw["train_labels"]      = pd.read_csv(DATA_DIR / "Train.csv")
    raw["test_ids"]          = pd.read_csv(DATA_DIR / "Test.csv")
    raw["variable_defs"]     = pd.read_csv(DATA_DIR / "VariableDefinitions.csv")
    raw["sample_submission"] = pd.read_csv(DATA_DIR / "SampleSubmission.csv")

    for k, v in raw.items():
        log.info(f"  {k}: {v.shape}")

    return raw


def clean_transactions(txn):
    log.info("Cleaning transaction data ...")

    # Parse date
    txn[DATE_COL] = pd.to_datetime(txn[DATE_COL], errors="coerce")

    # Handle both IsDebitCredit and DebitCredit column names
    dc_col = None
    for candidate in ["IsDebitCredit", DEBIT_CREDIT_COL, "DebitCredit"]:
        if candidate in txn.columns:
            dc_col = candidate
            break

    if dc_col is not None:
        raw_vals = txn[dc_col].astype(str).str.strip().str.upper()
        txn["debit_credit"] = np.where(
            raw_vals.isin(["D", "DEBIT", "-1", "DR", "1", "TRUE", "YES"]),
            "debit", "credit"
        )
        del raw_vals
        txn.drop(columns=[dc_col], inplace=True, errors="ignore")
    elif AMOUNT_COL in txn.columns:
        txn[AMOUNT_COL] = pd.to_numeric(txn[AMOUNT_COL], errors="coerce")
        txn["debit_credit"] = np.where(txn[AMOUNT_COL] < 0, "debit", "credit")

    # Absolute amount
    if AMOUNT_COL in txn.columns:
        txn[AMOUNT_COL]     = pd.to_numeric(txn[AMOUNT_COL], errors="coerce")
        txn[AMOUNT_ABS_COL] = txn[AMOUNT_COL].abs().astype(np.float32)
        txn.drop(columns=[AMOUNT_COL], inplace=True)

    # Drop heavy unused columns
    drop_cols = [c for c in [
        "StatementBalance", "ReversalTypeDescription",
        "AccountID", DEBIT_CREDIT_COL,
    ] if c in txn.columns]
    if drop_cols:
        txn.drop(columns=drop_cols, inplace=True)

    # Convert to category dtype to save RAM
    for col in ["debit_credit", "TransactionBatchDescription",
                "TransactionTypeDescription"]:
        if col in txn.columns:
            txn[col] = txn[col].astype("category")

    txn.dropna(subset=[DATE_COL], inplace=True)
    txn.dropna(axis=1, how="all", inplace=True)
    gc.collect()

    log.info(f"  Transactions cleaned: {txn.shape}")
    return txn


def clean_financials(fin):
    log.info("Cleaning financials ...")
    fin = fin.copy()

    date_cols = [c for c in fin.columns if "date" in c.lower() or "Date" in c]
    for dc in date_cols:
        fin[dc] = pd.to_datetime(fin[dc], errors="coerce")

    id_cols = {CUSTOMER_ID_COL, "AccountID"}
    for col in fin.columns:
        if col in id_cols or col in date_cols:
            continue
        if fin[col].dtype == object:
            continue
        fin[col] = pd.to_numeric(fin[col], errors="coerce").astype(np.float32)

    if date_cols and CUSTOMER_ID_COL in fin.columns:
        fin.sort_values([CUSTOMER_ID_COL, date_cols[0]], inplace=True)
        fin = fin.groupby(CUSTOMER_ID_COL, group_keys=False).apply(
            lambda g: g.ffill().bfill()
        )

    log.info(f"  Financials cleaned: {fin.shape}")
    return fin


def clean_demographics(demo):
    log.info("Cleaning demographics ...")
    demo = demo.copy()

    num_cols = demo.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        demo[num_cols] = SimpleImputer(strategy="median").fit_transform(
            demo[num_cols]
        )

    le = LabelEncoder()
    for col in list(demo.select_dtypes(include="object").columns):
        if col == CUSTOMER_ID_COL:
            continue
        if demo[col].nunique() <= 30:
            demo[f"{col}_enc"] = le.fit_transform(
                demo[col].fillna("Unknown")
            ).astype(np.int16)
        else:
            freq = demo[col].value_counts(normalize=True)
            demo[f"{col}_freq_enc"] = (
                demo[col].map(freq).fillna(0).astype(np.float32)
            )
        demo.drop(columns=[col], inplace=True)

    log.info(f"  Demographics cleaned: {demo.shape}")
    return demo