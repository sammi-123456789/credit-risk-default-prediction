# 📌 Credit Risk / Loan Default Decisioning System

An end-to-end credit risk pipeline built on **2.2M+ Lending Club loans** that predicts each applicant's **probability of default (PD)** and converts it into **Approve / Review / Reject** lending decisions using business-controlled thresholds.

The focus is not just prediction accuracy, but **business decisions**: models are compared on expected financial loss, and SHAP is used to explain what drives risk.

---

## 🎯 Business Problem

Lending involves asymmetric risk: approving a defaulter causes a large credit loss, while rejecting a good borrower means a smaller loss of revenue. The goal is to estimate default risk for each applicant and translate it into decisions aligned with the lender's risk appetite.

---

## 🧱 Pipeline

| Phase | What it does |
|---|---|
| **1. Data Ingestion** | Chunk-based loading of 2,260,701 raw loan records (151 columns) |
| **2. Cleaning & Leakage Prevention** | Kept only resolved loans (1,345,350; ~20% default rate), removed post-loan columns such as payments and recoveries, dropped features with >50% missing values |
| **3. Feature Engineering** | Interpretable risk signals: income-to-loan ratio, installment-to-income, debt-to-income (DTI), credit history length at loan issue, employment length, delinquency flag |
| **4. Modeling** | Logistic Regression (baseline), Random Forest, and XGBoost, all with the same class-imbalance handling; missing values imputed inside the pipeline using training data only |
| **5. Evaluation** | ROC-AUC, Precision–Recall, confusion matrices at business thresholds, cost-sensitive comparison |
| **6. Decision Engine** | Converts PDs into Approve / Review / Reject and analyzes default rates and cost per decision bucket |
| **7. Explainability** | SHAP analysis of the XGBoost model to identify key risk drivers |

---

## 📊 Results

### Ranking Quality (ROC-AUC on 25% hold-out test set)

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.707 |
| Random Forest | 0.706 |
| **XGBoost** | **0.717** |

### Decision Engine: Default Rate by Bucket

| Model | Approve | Review | Reject |
|---|---|---|---|
| Logistic Regression | 8.7% | 17.2% | 30.9% |
| Random Forest | 8.4% | 16.7% | 30.4% |
| **XGBoost** | 8.6% | 17.2% | **31.8%** |

Against a portfolio baseline of ~20%, all models produce correctly ordered buckets: approved loans default at less than half the average rate, while rejected loans default at roughly 1.5×.

### Expected Business Cost (illustrative costs: ₹100,000 per approved defaulter, ₹10,000 per rejected good borrower)

| Model | Total Expected Cost |
|---|---|
| Logistic Regression | ₹2,125.8M |
| **Random Forest** | **₹2,083.1M** |
| XGBoost | ₹2,129.5M |

---

## 🔎 Key Findings

- **XGBoost ranks risk best, but Random Forest is cheapest at the fixed thresholds.** ROC-AUC measures separation across all thresholds, while cost is measured at one fixed threshold. Since each model's probabilities sit on a different scale, the same threshold is not optimal for every model, so thresholds should be tuned per model.
- **SHAP confirms the model learned sensible patterns:** interest rate is the strongest driver, 60-month loans are riskier than 36-month loans, better grades reduce risk, and past delinquencies increase it.
- **Debt-to-income ratio is the 4th most influential feature**, confirming the value of adding it during feature engineering.
- **Interest rate partly reflects Lending Club's own risk grading**, so the model builds on, rather than replaces, the lender's existing assessment.

---

## 🛠️ Improvements Made in This Version

- Credit history length now measured at the loan issue date (previously measured against a fixed 2018 date)
- Added debt-to-income ratio and employment length as features
- Moved missing-value imputation into the modeling pipeline to prevent test-set leakage
- Applied the same class-imbalance handling to all models for a fair comparison (previously only Logistic Regression was balanced, which made the cost comparison misleading)
- Added XGBoost and SHAP explainability

---

## 🔮 Limitations & Future Work

- The current thresholds reject ~42–45% of applicants, which is higher than a typical lender would accept; thresholds should be tuned per model against a target approval rate.
- Use an out-of-time validation split (train on older loans, test on newer ones) to better reflect real deployment.
- Replace flat cost assumptions with loan-level expected loss (PD × LGD × EAD).
- Calibrate predicted probabilities so PDs can be interpreted as true default likelihoods.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP

## ▶️ How to Run

The notebook is designed to run on Kaggle due to the dataset size (~2.5 GB). Open it on Kaggle, attach the [Lending Club dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club), and select **Run All**.
