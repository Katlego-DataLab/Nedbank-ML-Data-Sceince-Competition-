import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR, TARGET

log = logging.getLogger("nedbank.diagnostics")

plt.rcParams.update({
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def plot_feature_importance(xgb_models, lgb_models, feature_names,
                             top_n=30, save_path=OUTPUT_DIR / "feature_importance.png"):
    xgb_imp = np.mean([m.feature_importances_ for m in xgb_models], axis=0)
    lgb_imp = np.mean([m.feature_importances_ for m in lgb_models], axis=0)

    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "xgb_imp":    xgb_imp,
        "lgb_imp":    lgb_imp,
        "avg_imp":    0.5 * xgb_imp + 0.5 * lgb_imp,
    }).sort_values("avg_imp", ascending=False)

    # Save importance as CSV (downloadable diagnostic)
    csv_path = OUTPUT_DIR / "feature_importance.csv"
    fi_df.to_csv(csv_path, index=False)
    log.info(f"Feature importance CSV saved: {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, top_n // 2)))
    for ax, imp_col, label, color in [
        (axes[0], "xgb_imp", "XGBoost",  "#2563EB"),
        (axes[1], "lgb_imp", "LightGBM", "#10B981"),
    ]:
        top = fi_df.nlargest(top_n, imp_col)
        sns.barplot(data=top, x=imp_col, y="feature", ax=ax, color=color)
        ax.set_title(f"Top {top_n} Features — {label}")
        ax.set_xlabel("Mean Gain")
        ax.set_ylabel("")

    plt.suptitle("Feature Importances (avg across CV folds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Feature importance chart saved: {save_path}")
    return fi_df


def plot_prediction_distribution(y_true, y_pred, label="OOF",
                                  save_path=OUTPUT_DIR / "prediction_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, c="#2563EB", rasterized=True)
    lim = max(float(y_true.max()), float(y_pred.max()))
    axes[0].plot([0, lim], [0, lim], "r--", lw=1.5)
    axes[0].set_xlabel(f"Actual {TARGET}")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Actual vs Predicted ({label})")

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60, color="#10B981", edgecolor="white")
    axes[1].axvline(0, color="red", lw=1.5)
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_title(f"Residual Distribution ({label})")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Prediction distribution chart saved: {save_path}")


def plot_cv_scores(fold_metrics_xgb, fold_metrics_lgb,
                   save_path=OUTPUT_DIR / "cv_scores.png"):
    xgb_rmsle = [m["RMSLE"] for m in fold_metrics_xgb]
    lgb_rmsle = [m["RMSLE"] for m in fold_metrics_lgb]
    folds     = list(range(1, len(xgb_rmsle) + 1))

    # Save CV scores as CSV
    cv_df = pd.DataFrame({
        "fold":      folds,
        "xgb_rmsle": xgb_rmsle,
        "lgb_rmsle": lgb_rmsle,
        "xgb_rmse":  [m["RMSE"] for m in fold_metrics_xgb],
        "lgb_rmse":  [m["RMSE"] for m in fold_metrics_lgb],
    })
    csv_path = OUTPUT_DIR / "cv_scores.csv"
    cv_df.to_csv(csv_path, index=False)
    log.info(f"CV scores CSV saved: {csv_path}")

    x     = np.arange(len(folds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, xgb_rmsle, width, label="XGBoost",  color="#2563EB")
    ax.bar(x + width / 2, lgb_rmsle, width, label="LightGBM", color="#10B981")
    ax.axhline(np.mean(xgb_rmsle), color="#2563EB", linestyle="--", alpha=0.7,
               label=f"XGB mean={np.mean(xgb_rmsle):.4f}")
    ax.axhline(np.mean(lgb_rmsle), color="#10B981", linestyle="--", alpha=0.7,
               label=f"LGB mean={np.mean(lgb_rmsle):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("RMSLE")
    ax.set_title("Cross-validation RMSLE by Fold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"CV scores chart saved: {save_path}")


def plot_segment_distribution(Nedbank_data,
                               save_path=OUTPUT_DIR / "segment_distribution.png"):
    if "customer_segment" not in Nedbank_data.columns:
        return
    counts = Nedbank_data["customer_segment"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("Set2", len(counts)))
    axes[0].set_title("Customer Segment Distribution (Pie)")
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[1], palette="Set2")
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Customer Segment Distribution (Bar)")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Segment distribution chart saved: {save_path}")


def print_cv_summary(fold_metrics_xgb, fold_metrics_lgb):
    header = f"{'Fold':<6} {'XGB RMSLE':>10} {'LGB RMSLE':>10} {'XGB RMSE':>10} {'LGB RMSE':>10}"
    sep    = "-" * len(header)
    log.info("=" * len(header))
    log.info(header)
    log.info(sep)
    for i, (xm, lm) in enumerate(zip(fold_metrics_xgb, fold_metrics_lgb)):
        log.info(
            f"{i+1:<6} {xm['RMSLE']:>10.4f} {lm['RMSLE']:>10.4f} "
            f"{xm['RMSE']:>10.2f} {lm['RMSE']:>10.2f}"
        )
    log.info(sep)
    log.info(
        f"{'Mean':<6} "
        f"{np.mean([m['RMSLE'] for m in fold_metrics_xgb]):>10.4f} "
        f"{np.mean([m['RMSLE'] for m in fold_metrics_lgb]):>10.4f} "
        f"{np.mean([m['RMSE']  for m in fold_metrics_xgb]):>10.2f} "
        f"{np.mean([m['RMSE']  for m in fold_metrics_lgb]):>10.2f}"
    )
    log.info("=" * len(header))
