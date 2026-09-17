# loan-default-prediction

ML pipeline predicting loan default risk using real Kaggle lending data — Logistic Regression, Decision Tree & Random Forest with cost-benefit and risk-tiering analysis.

Predicting whether a loan applicant is likely to default, using **two real, publicly available Kaggle lending datasets** combined into one larger real-world dataset (~170,000 rows). This replaces an earlier version of the project that used synthetic (randomly generated) data.

```
Filtering → Dataset Create → Realistic Check → Clean & Preprocess
   → Feature Engineering → Feature Selection → Train/Test Split (70/30)
   → Train 3 Models → Evaluate → Cross-Validation → Class Imbalance Handling
   → Feature Importance → Predict New Applicant → Cost-Benefit Analysis
   → Risk Tiering → Executive Summary → Regulatory Note
```

## Business Problem

Lenders need a consistent way to assess credit risk before approving a loan. This project builds and compares three classification models that estimate an applicant's probability of default, then translates those predictions into business-relevant outputs: risk tiers, an estimated dollar cost of prediction errors, and a plain-language summary for non-technical stakeholders.

## Data Sources

| Dataset | Description | Size |
|---|---|---|
| `credit_risk_dataset.csv` | Kaggle credit risk dataset | 32,581 borrowers, 12 columns |
| `Loan_Default_csv.xlsx` | Kaggle loan default dataset | 34 columns (subset used) |

The two datasets use different column names, units, and encodings (e.g. letter loan grades vs. numeric credit scores, monthly vs. annual income, text age bands vs. numeric age). They are harmonized onto one shared schema and combined — see the **Dataset Create** section of the notebook for the full mapping.

> Datasets are not included in this repository due to size/licensing. Download both from Kaggle and place them in the project root (or update the file paths in the notebook) before running.

## Pipeline

1. **Filtering** — remove invalid/impossible rows (e.g. negative income, unrealistic ages) and drop mostly-empty or irrelevant columns.
2. **Dataset Create** — harmonize both datasets onto a shared schema and combine them.
3. **Realistic Check** — sanity-check default rate, ranges, and source balance in the combined data.
4. **Clean & Preprocess** — missing-value flags, median imputation, outlier clipping (1st/99th percentile).
5. **Feature Engineering** — `loan_to_income_ratio`, `credit_score_normalized`, `high_dti_flag`, `age_group`.
6. **Feature Selection** — correlation with target + Random Forest feature importance; `source` is excluded to prevent the model from learning which dataset a row came from.
7. **Train/Test Split** — stratified 70/30 split, reused across all models.
8. **Train 3 Models** — Logistic Regression, Decision Tree, Random Forest.
9. **Evaluate** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves.
10. **Cross-Validation** — 5-fold stratified CV to confirm results are stable.
11. **Class Imbalance Handling** — retrain all models with `class_weight="balanced"` (only ~24% of applicants defaulted).
12. **Feature Importance** — compare what drives each model's predictions.
13. **Predict New Applicant** — helper function converts raw, human-readable applicant data into model-ready features; includes an interactive `ipywidgets` demo.
14. **Cost-Benefit Analysis** — translates false negatives/positives into estimated dollar impact.
15. **Risk Tiering** — buckets applicants into Low / Medium / High risk based on predicted default probability.
16. **Executive Summary & Regulatory Note** — plain-language summary plus a fair-lending/RBI compliance note.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | 0.697 |
| Decision Tree | — | — | — | — | 0.978 |
| Random Forest | — | — | — | — | 0.979 |

*(See the notebook's Section 12 output for exact values; Decision Tree and Random Forest substantially outperform Logistic Regression on this dataset.)*

**Top predictive features:** `interest_rate` dominates both tree-based models, followed at a distance by `debt_to_income_pct` and `loan_to_income_ratio`.

## Repository Structure

```
.
├── loan_default_ml_models_real_data.ipynb   # Main analysis notebook
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/loan-default-prediction.git
cd loan-default-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data
Download `credit_risk_dataset.csv` and `Loan_Default_csv.xlsx` from Kaggle and place them in the project root.

### 4. Run the notebook
```bash
jupyter notebook loan_default_ml_models_real_data.ipynb
```
Or open it directly in Google Colab (upload the two CSV/XLSX files to the Colab session when prompted).

## Tech Stack

- Python, pandas, numpy
- scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- matplotlib, seaborn (visualization)
- ipywidgets (interactive applicant demo)

## Regulatory & Fairness Note

This notebook intentionally excludes protected characteristics from the feature set. Disparate-impact testing across demographic groups was **not** performed and would be a required next step before any real-world lending use, along with the transparency and human-oversight expectations outlined in RBI algorithmic lending guidance (the data reflects an Indian lending context).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
