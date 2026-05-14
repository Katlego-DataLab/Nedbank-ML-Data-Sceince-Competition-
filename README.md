# 🏦 Nedbank · Banking on Behaviour
### *Predicting Customer Transaction Activity to Drive Retention & Revenue*
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Poisson-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Ensemble-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CatBoost-Calibrated-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Competition-Zindi-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>
<p align="center">
  <a href="https://nedbank-customer-engagement.streamlit.app/">
    <img src="https://img.shields.io/badge/Open%20in%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open in Streamlit"/>
  </a>
</p>

---

##  Quick Overview

> Built a **two-stage ML pipeline** that processes **18 million+ raw banking transactions** and predicts how many transactions each customer will make in the next 3 months enabling Nedbank to act on churn risk, tailor product offers, and optimise marketing spend before customers disengage.

| What | Result |
|------|--------|
|  Target | Predict next-3-month transaction count per customer |
|  Data Scale | **18,017,073 raw transactions** processed end-to-end |
|  Best CV Score | **RMSLE ~0.35** (XGB + LGB ensemble, temporal CV) |
|  Models | XGBoost · LightGBM · CatBoost · Two-stage zero-inflation |
|  Business Value | Proactive churn intervention, personalised engagement, resource planning |
|  Deployed | [Live Streamlit Dashboard](https://nedbank-customer-engagement.streamlit.app/) |

---

##  Business Problem

Banks lose millions every year to **silent churn**  customers who slowly stop transacting without ever formally closing their account. By the time a relationship manager notices, the customer has already moved to a competitor.

This project answers one critical question:

> **"Which customers are likely to go quiet in the next quarter and how active will the rest be?"**

Getting this right means:
-  **Revenue protection** — retain high-value customers before they leave
-  **Smarter marketing** — stop wasting budget on already-engaged customers
-  **Resource allocation** — focus retention teams where they'll have the most impact
-  **Early warning** — flag at-risk customers 90 days in advance, not after the fact

---

##  Business Impact

**Who uses this:**
-  **Marketing teams** — segment customers into Active / At Risk / Dormant buckets for targeted campaigns
-  **Retention / CRM teams** — prioritise outreach to customers predicted to drop off
-  **Finance & planning** — forecast transaction volumes per segment for capacity planning
-  **Relationship managers** — receive alerts for at-risk clients in their portfolio

**What decisions this enables:**
- Trigger a personalised offer 90 days before a customer goes quiet
- Rank customers by expected activity to assign relationship manager bandwidth
- Build "next best action" workflows in the bank's CRM

---

##  Key Results

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Raw Transactions Processed** | **18,017,073** | Full-scale production data, not a sample |
| **CV RMSLE (XGB)** | ~0.35 | Predictions within ~35% error on log scale robust for count data |
| **CV RMSLE (LGB)** | ~0.36 | Strong second model; divergence confirms ensemble diversity |
| **Ensemble Gain** | ✅ | Blended predictions outperform either model alone |
| **Zero-inflation Handling** | ✅ | Stage-1 classifier explicitly flags customers likely to be inactive |
| **Features Engineered** | 100+ | Velocity, decay, debit/credit, balance, behavioural embeddings |

> **Why are these results credible?**
> Temporal cross-validation was used, future data never leaked into training folds. The two-stage approach (activity classifier × count regressor) explicitly handles the large proportion of zero-transaction customers, which is the hardest part of this problem.

---

##  Live Dashboard

**👉 [nedbank-customer-engagement.streamlit.app](https://nedbank-customer-engagement.streamlit.app/)**

The deployed Streamlit app allows business users to:
- 🔍 **Search any customer ID** and see their predicted next-quarter activity score
- 🟢🟡🔴 **View engagement status** — Active / At Risk / Dormant with confidence scores
- 📊 **Explore segment breakdowns** — pie charts, prediction distributions, threshold analysis
- 📈 **Review model performance** — CV RMSLE scores and top predictive features
- 📥 **Download a prioritised outreach list** filtered by segment and risk level

```bash
# To run locally:
streamlit run app.py
```

---

## ⚙️ Solution Overview

```
18,017,073 Raw Transactions + Financials + Demographics
        │
        ▼
┌─────────────────────────────────────────────────────┐
│             Feature Engineering                     │
│  Temporal · Velocity · Decay · Debit/Credit         │
│  Balance Trajectory · Type Embeddings · Anomaly     │
│  Segmentation (KMeans/PCA) · Holiday Granularity    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           Feature Selection                         │
│  Variance filter → Correlation filter →             │
│  Rank by |corr(log1p(target))| → Top 100            │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐    ┌────────────────────────┐
│  Stage 1 Classifier │    │  Stage 2 Regressors    │
│  (XGB Binary)       │  × │  XGBoost (Poisson)     │
│  P(customer active) │    │  LightGBM (Poisson)    │
│                     │    │  CatBoost (RMSE)       │
└─────────────────────┘    └────────────────────────┘
        │                           │
        └──────────── × ────────────┘
                      │
              Isotonic Calibration
                      │
              Optimised Ensemble Blend
                      │
        predictions.csv → Streamlit Dashboard
```

---

##  Key Technical Decisions

| Decision | Why It Matters |
|----------|---------------|
| **Poisson objective** | Transaction counts are non-negative integers — Poisson loss is a better fit than RMSE |
| **Temporal CV** | Training folds always precede validation folds — prevents future-data leakage |
| **Two-stage zero-inflation** | Stage 1 explicitly flags inactive customers before Stage 2 predicts counts |
| **Isotonic calibration** | OOF predictions recalibrated to correct systematic over/under-prediction bias |
| **Correlation-based feature selection** | Ranked by `|corr(log1p(y))|` — directly aligned with RMSLE |
| **Decay-weighted aggregations** | Half-lives of 30/90/180 days capture short, medium, and long memory |
| **Memory-efficient processing** | 18M+ rows processed in chunks — never loading the full dataset into memory at once |

---

##  Key Insights

- **Recency dominates.** How recently a customer transacted is a stronger predictor than their all-time average.
- **Holiday spend signals future activity.** Customers with elevated Nov/Dec/Jan transaction spikes tend to be consistently engaged year-round.
- **Debit/credit ratio is a behaviour fingerprint.** Net spenders have different future activity patterns than net savers.
- **Zero is the hardest prediction.** ~30–40% of customers transact rarely or not at all — the two-stage model was built specifically to address this.
- **Balance volatility predicts disengagement.** Customers with erratic end-of-month balance trajectories show higher churn risk.

---

##  Production Thinking

Here's how this pipeline would operate inside a real bank:

```
Monthly Data Pull (Core Banking System)
    → Feature pipeline runs overnight (Airflow DAG)
    → Predictions written to CRM database
    → Segment flags updated (Active / At Risk / Dormant)
    → Alerts triggered for customers crossing "at-risk" threshold
    → Relationship manager dashboard refreshed (Streamlit live)
    → Model performance logged (actual vs predicted, 3-month lag)
    → Retraining triggered quarterly or when RMSLE degrades > 10%
```

**Integration points:**
- 🗃️ **SQLite / Parquet feature store** (already implemented) → upgrade to Snowflake / BigQuery
- 📊 **Streamlit dashboard** — [live and deployed](https://nedbank-customer-engagement.streamlit.app/)
- 🔔 **CRM alerts** — predictions feed into Salesforce / Dynamics via API
- 🔁 **Retraining cadence** — quarterly with temporal walk-forward validation

---

##  Future Improvements

- [ ] **Add Prophet seasonality features** — already scaffolded in pipeline
- [ ] **Customer lifetime value integration** — weight predictions by revenue tier
- [ ] **Real-time scoring API** — wrap model in FastAPI for event-triggered predictions
- [ ] **Fairness audit** — check predictions don't systematically disadvantage demographic groups
- [ ] **Graph features** — model peer-group behaviour across transaction channels

---

##  Project Structure

```
📦 Nedbank-Customer-Behaviour-app/
├── 📄 app.py                      ← Streamlit dashboard (deployed)
├── 📄 main_v3.py                  ← Pipeline entry point
├── 📄 feature_engineering_v3.py   ← All feature construction
├── 📄 feature_selection_v5.py     ← Pre-modelling feature filter
├── 📄 modelling_v3.py             ← XGB / LGB / CatBoost + two-stage
├── 📄 metrics.py                  ← RMSLE, RMSE, MAE evaluation
├── 📄 diagnostics.py              ← Charts and CV summaries
├── 📄 store.py                    ← SQLite + Parquet feature store
├── 📄 config.py                   ← All constants and paths
├── 📄 config_additions_v3.py      ← v3-specific config additions
├── 📄 data_loader.py              ← Raw data ingestion & cleaning
├── 📄 predictions.csv             ← Customer-level predictions
├── 📄 cv_scores.csv               ← Cross-validation results
├── 📄 feature_importance.csv      ← Top features by model
└── 📄 requirements.txt
```

---

##  How to Run

```bash
# 1. Clone
git clone https://github.com/Katlego-DataLab/Nedbank-Customer-Behaviour-app.git
cd Nedbank-Customer-Behaviour-app

# 2. Install
pip install -r requirements.txt

# 3. Run pipeline (after adding data files)
python main_v3.py --no-tune --no-prophet

# 4. Launch dashboard
streamlit run app.py
```

Or skip all of this and visit the **[live dashboard](https://nedbank-customer-engagement.streamlit.app/)** directly.

---

##  About This Project

This is part of my **12-month journey from aspiring data scientist to ML practitioner**. Each project tackles a real business problem, uses production-grade code structure, and is built to be deployable — not just notebook-level exploratory work.

**Skills demonstrated:** Large-scale data processing (18M+ rows) · Feature engineering · Ensemble modelling · Time series · Temporal validation · Business framing · Production pipeline design · Streamlit deployment

---

<p align="center">
  <i>Built with 🤍 as part of a 12-month Data Science portfolio journey</i><br/>
  <a href="https://katlego-datalab.github.io/Website-updated-/">Portfolio</a> ·
  <a href="https://www.linkedin.com/in/katlego-mathebula-044a703b4">LinkedIn</a> ·
  <a href="https://nedbank-customer-engagement.streamlit.app/">Live Dashboard</a>
</p>
