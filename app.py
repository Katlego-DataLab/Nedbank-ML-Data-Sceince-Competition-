import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Colour palette ──────────────────────────────────────────────
CHARCOAL    = "#2C2C2C"
GREY        = "#6B6B6B"
LIGHT_GREY  = "#E8E6E1"
BEIGE       = "#C8B89A"
IVORY       = "#F5F0E8"
TERRACOTTA  = "#C4622D"
TERRA_LIGHT = "#F0D5C8"
ARMY_GREEN  = "#4A5240"
ARMY_LIGHT  = "#C8D0B8"
WHITE       = "#FFFFFF"

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Nedbank · Customer Engagement Radar",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{
        background-color: {IVORY};
        color: {CHARCOAL};
    }}
    [data-testid="stSidebar"] {{
        background-color: {CHARCOAL} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {IVORY} !important;
    }}
    html, body, [class*="css"] {{
        font-family: 'Georgia', serif;
        color: {CHARCOAL};
    }}
    [data-testid="metric-container"] {{
        background-color: {WHITE};
        border: 1px solid {BEIGE};
        border-radius: 12px;
        padding: 16px;
    }}
    [data-testid="metric-container"] label {{
        color: {GREY} !important;
        font-size: 13px !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {CHARCOAL} !important;
        font-size: 28px !important;
        font-weight: 600 !important;
    }}
    h1 {{ color: {CHARCOAL}; font-family: Georgia, serif; border-bottom: 2px solid {TERRACOTTA}; padding-bottom: 8px; }}
    h2 {{ color: {ARMY_GREEN}; font-family: Georgia, serif; }}
    h3 {{ color: {TERRACOTTA}; font-family: Georgia, serif; }}
    hr {{ border-color: {BEIGE}; }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {BEIGE};
        border-radius: 8px;
    }}
    .stDownloadButton > button {{
        background-color: {ARMY_GREEN} !important;
        color: {IVORY} !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 14px !important;
    }}
    .stDownloadButton > button:hover {{
        background-color: {TERRACOTTA} !important;
    }}
    .stTextInput > div > div > input {{
        border: 1.5px solid {BEIGE} !important;
        border-radius: 8px !important;
        background-color: {WHITE} !important;
        color: {CHARCOAL} !important;
        font-size: 15px !important;
    }}
    .stMultiSelect > div {{
        border: 1.5px solid {BEIGE} !important;
        border-radius: 8px !important;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }}
    .badge-active    {{ background: {ARMY_LIGHT};  color: {ARMY_GREEN}; }}
    .badge-risk      {{ background: #F5E6C8;        color: #7A5C00; }}
    .badge-dormant   {{ background: {TERRA_LIGHT};  color: {TERRACOTTA}; }}
    .info-box {{
        background-color: {WHITE};
        border-left: 4px solid {TERRACOTTA};
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 14px;
        color: {CHARCOAL};
    }}
</style>
""", unsafe_allow_html=True)


# ── Matplotlib theme ─────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  IVORY,
        "axes.facecolor":    WHITE,
        "axes.edgecolor":    BEIGE,
        "axes.labelcolor":   CHARCOAL,
        "xtick.color":       GREY,
        "ytick.color":       GREY,
        "text.color":        CHARCOAL,
        "grid.color":        LIGHT_GREY,
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "font.family":       "serif",
    })

set_plot_style()

PALETTE = [TERRACOTTA, ARMY_GREEN, BEIGE, CHARCOAL, GREY]


# ── Data loader ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    pred_path = os.path.join(base, "predictions.csv")
    cv_path   = os.path.join(base, "cv_scores.csv")
    fi_path   = os.path.join(base, "feature_importance.csv")

    if not os.path.exists(pred_path):
        st.error("predictions.csv not found. Run your pipeline first.")
        st.stop()

    pred = pd.read_csv(pred_path)
    raw_col = "predicted_next_3m_txn_count_raw"

    # ── Fixed business-logic thresholds ─────────────────────────
    # 71 % of predictions are tied at 145.31 (model degeneracy),
    # so percentile-based bins collapse to one bucket.
    # These hard thresholds are meaningful given the data range
    # (min ≈ 5, median ≈ 145, max ≈ 922).
    DORMANT_MAX = 50    # fewer than 50 predicted txns → Dormant
    RISK_MAX    = 145   # 50–145 → At Risk  (just below the degenerate spike)
                        # > 145  → Active

    pred["risk_flag"] = pd.cut(
        pred[raw_col],
        bins=[-np.inf, DORMANT_MAX, RISK_MAX, np.inf],
        labels=["Dormant", "At Risk", "Active"],
    ).astype(str)

    p33 = DORMANT_MAX   # reuse these names so the rest of the app still works
    p66 = RISK_MAX

    cv = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
    fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    return pred, cv, fi, p33, p66

    # ── Dynamic risk bins based on actual data distribution ──────
    raw_col = "predicted_next_3m_txn_count_raw"
    p33 = pred[raw_col].quantile(0.33)
    p66 = pred[raw_col].quantile(0.66)

    # If percentiles are identical (many tied values), use rank-based assignment
    if p33 == p66:
        pred["risk_flag"] = pd.qcut(
            pred[raw_col].rank(method="first"),
            q=3,
            labels=["Dormant", "At Risk", "Active"]
        ).astype(str)
        p33 = pred.loc[pred["risk_flag"] == "Dormant", raw_col].max()
        p66 = pred.loc[pred["risk_flag"] == "At Risk",  raw_col].max()
    else:
        pred["risk_flag"] = pd.cut(
            pred[raw_col],
            bins=[-np.inf, p33, p66, np.inf],
            labels=["Dormant", "At Risk", "Active"]
        ).astype(str)

    cv = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
    fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    return pred, cv, fi, p33, p66


df, cv, fi, p33, p66 = load_data()

active_n  = (df["risk_flag"] == "Active").sum()
risk_n    = (df["risk_flag"] == "At Risk").sum()
dormant_n = (df["risk_flag"] == "Dormant").sum()
total_n   = len(df)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Nedbank")
    st.markdown("### Customer Engagement Radar")
    st.markdown("---")
    st.markdown(f"**Total customers:** {total_n:,}")
    st.markdown(f"🟢 Active:  {active_n:,}")
    st.markdown(f"🟡 At Risk: {risk_n:,}")
    st.markdown(f"🔴 Dormant: {dormant_n:,}")
    st.markdown("---")
    st.markdown(f"**Risk thresholds** (percentile-based)")
    st.markdown(f"🔴 Dormant: < {p33:.0f} txns")
    st.markdown(f"🟡 At Risk: {p33:.0f} – {p66:.0f} txns")
    st.markdown(f"🟢 Active:  > {p66:.0f} txns")
    st.markdown("---")
    st.markdown("**Project:** Banking on Behaviour")
    st.markdown("**Model:** XGBoost · LightGBM · CatBoost")
    st.markdown("**Metric:** RMSLE ~0.35")
    st.markdown("---")
    st.caption("Built as part of a 12-month Data Science portfolio journey.")


# ══════════════════════════════════════════════════════════════════
# MAIN PAGE TITLE
# ══════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Nedbank · Customer Engagement Radar")
st.markdown(
    "Predicting next-quarter transaction activity to drive proactive retention, "
    "targeted marketing, and smarter resource allocation."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — KPI CARDS
# ══════════════════════════════════════════════════════════════════
st.markdown("##  Portfolio Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers",  f"{total_n:,}")
c2.metric("🟢 Active",        f"{active_n:,}",  f"{active_n/total_n*100:.1f}%")
c3.metric("🟡 At Risk",       f"{risk_n:,}",    f"{risk_n/total_n*100:.1f}%")
c4.metric("🔴 Dormant",       f"{dormant_n:,}", f"{dormant_n/total_n*100:.1f}%")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — CHARTS ROW 1
# ══════════════════════════════════════════════════════════════════
st.markdown("##  Engagement Distribution")

col_left, col_right = st.columns(2)

# ── Pie chart (fixed) ────────────────────────────────────────────
with col_left:
    st.markdown("### Customer Risk Segments")
    fig1, ax1 = plt.subplots(figsize=(5, 4.5))

    sizes  = [dormant_n, risk_n, active_n]
    colors = [TERRACOTTA, BEIGE, ARMY_GREEN]
    labels = [
        f"Dormant\n{dormant_n:,} ({dormant_n/total_n*100:.1f}%)",
        f"At Risk\n{risk_n:,} ({risk_n/total_n*100:.1f}%)",
        f"Active\n{active_n:,} ({active_n/total_n*100:.1f}%)",
    ]

    # Explode small slices so they're visible
    explode = [0.05 if s / total_n < 0.05 else 0 for s in sizes]

    wedges, texts = ax1.pie(
        sizes,
        labels=None,           # we'll use a legend instead
        colors=colors,
        startangle=90,
        explode=explode,
        wedgeprops={"edgecolor": WHITE, "linewidth": 2},
        pctdistance=0.75,
    )

    # Clean legend instead of overlapping labels
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(colors, labels)
    ]
    ax1.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        fontsize=9,
        frameon=False,
    )

    ax1.set_facecolor(IVORY)
    fig1.patch.set_facecolor(IVORY)
    plt.tight_layout()
    st.pyplot(fig1)

# ── Histogram (fixed) ────────────────────────────────────────────
with col_right:
    st.markdown("### Predicted Transaction Count Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))

    raw_col = "predicted_next_3m_txn_count_raw"
    # Clip at 95th percentile so the chart doesn't pile up at one edge
    p95     = df[raw_col].quantile(0.95)
    clipped = df[raw_col].clip(0, p95)

    ax2.hist(clipped, bins=40, color=TERRACOTTA, edgecolor=WHITE, linewidth=0.5)
    ax2.set_xlabel("Predicted Transactions (next 3 months)", fontsize=11)
    ax2.set_ylabel("Number of Customers", fontsize=11)

    # Mean line
    mean_val = clipped.mean()
    ax2.axvline(mean_val, color=ARMY_GREEN, linestyle="--",
                linewidth=1.5, label=f"Mean: {mean_val:.1f}")

    # Threshold lines
    ax2.axvline(p33, color=TERRACOTTA, linestyle=":",
                linewidth=1.2, label=f"Dormant / At Risk: {p33:.0f}")
    ax2.axvline(p66, color=BEIGE, linestyle=":",
                linewidth=1.2, label=f"At Risk / Active: {p66:.0f}")

    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)
    fig2.patch.set_facecolor(IVORY)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — CV SCORES
# ══════════════════════════════════════════════════════════════════
if cv is not None:
    st.markdown("## Model Cross-Validation Performance")

    col_cv1, col_cv2 = st.columns([2, 1])

    with col_cv1:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        x     = np.arange(len(cv))
        width = 0.35
        ax3.bar(x - width/2, cv["xgb_rmsle"], width,
                label="XGBoost",  color=TERRACOTTA, edgecolor=WHITE)
        ax3.bar(x + width/2, cv["lgb_rmsle"], width,
                label="LightGBM", color=ARMY_GREEN,  edgecolor=WHITE)
        ax3.axhline(cv["xgb_rmsle"].mean(), color=TERRACOTTA,
                    linestyle="--", alpha=0.7, linewidth=1.2,
                    label=f"XGB mean {cv['xgb_rmsle'].mean():.4f}")
        ax3.axhline(cv["lgb_rmsle"].mean(), color=ARMY_GREEN,
                    linestyle="--", alpha=0.7, linewidth=1.2,
                    label=f"LGB mean {cv['lgb_rmsle'].mean():.4f}")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Fold {i+1}" for i in x])
        ax3.set_ylabel("RMSLE", fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, axis="y", alpha=0.4)
        fig3.patch.set_facecolor(IVORY)
        st.pyplot(fig3)

    with col_cv2:
        st.markdown("### CV Summary")
        st.markdown(f"""
        <div class="info-box">
            <b>XGBoost</b><br>
            Mean RMSLE: <b>{cv['xgb_rmsle'].mean():.4f}</b><br>
            Std: {cv['xgb_rmsle'].std():.4f}<br><br>
            <b>LightGBM</b><br>
            Mean RMSLE: <b>{cv['lgb_rmsle'].mean():.4f}</b><br>
            Std: {cv['lgb_rmsle'].std():.4f}<br><br>
            <i>Lower RMSLE = better prediction accuracy</i>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
if fi is not None:
    st.markdown("##  Top Predictive Features")
    top_fi = fi.nlargest(15, "avg_imp")

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.barh(
        top_fi["feature"][::-1],
        top_fi["avg_imp"][::-1],
        color=[TERRACOTTA if i % 2 == 0 else ARMY_GREEN
               for i in range(len(top_fi))],
        edgecolor=WHITE, linewidth=0.5
    )
    ax4.set_xlabel("Average Feature Importance (Gain)", fontsize=11)
    ax4.grid(True, axis="x", alpha=0.4)
    fig4.patch.set_facecolor(IVORY)
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — CUSTOMER LOOKUP
# ══════════════════════════════════════════════════════════════════
st.markdown("##  Customer Lookup")
st.markdown("Search any customer ID to see their predicted engagement and risk status.")

search_id = st.text_input("Enter Customer ID", placeholder="e.g. 100023")

if search_id:
    id_col = df.columns[0]
    match  = df[df[id_col].astype(str) == search_id.strip()]

    if len(match) == 0:
        st.error("Customer ID not found. Please check and try again.")
    else:
        row  = match.iloc[0]
        flag = str(row["risk_flag"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Customer ID",            str(row[id_col]))
        m2.metric("Predicted Transactions", f"{row['predicted_next_3m_txn_count_raw']:.1f}")
        m3.metric("Risk Status",            flag)

        if flag == "Dormant":
            badge_class = "badge-dormant"
            msg = "This customer is predicted to be <b>inactive</b> next quarter. Consider a re-engagement campaign."
        elif flag == "At Risk":
            badge_class = "badge-risk"
            msg = "This customer shows <b>declining engagement</b>. A proactive offer may help retain them."
        else:
            badge_class = "badge-active"
            msg = "This customer is predicted to remain <b>actively engaged</b> next quarter."

        st.markdown(f"""
        <div class="info-box">
            <span class="badge {badge_class}">{flag}</span><br><br>
            {msg}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — AT-RISK LIST
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🚨 At-Risk Customer List")
st.markdown("Export this list for your retention or CRM team.")

risk_options = st.multiselect(
    "Filter by risk level",
    options=["Dormant", "At Risk", "Active"],
    default=["Dormant", "At Risk"]
)

filtered = df[df["risk_flag"].isin(risk_options)].sort_values(
    "predicted_next_3m_txn_count_raw"
).reset_index(drop=True)

id_col = df.columns[0]
display_cols = [id_col, "predicted_next_3m_txn_count_raw",
                "predicted_next_3m_txn_count_log1p", "risk_flag"]
display_cols = [c for c in display_cols if c in filtered.columns]

st.markdown(f"**{len(filtered):,} customers match your filter.**")
st.dataframe(
    filtered[display_cols].rename(columns={
        "predicted_next_3m_txn_count_raw":   "Predicted Count (Raw)",
        "predicted_next_3m_txn_count_log1p": "Predicted Count (log1p)",
        "risk_flag":                         "Risk Status",
    }),
    use_container_width=True,
    height=380,
)

csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download At-Risk List as CSV",
    data=csv_bytes,
    file_name="at_risk_customers.csv",
    mime="text/csv",
)

st.markdown("---")

# ── Footer ───────────────────────────────────────────────────────
st.markdown(
    f"<p style='text-align:center; color:{GREY}; font-size:13px;'>"
    "Built with ❤️ · By Katlego Mathebula · "
    "Nedbank Customer Behaviour</p>",
    unsafe_allow_html=True,
) 
