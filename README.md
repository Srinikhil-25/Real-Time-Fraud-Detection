# Real-Time Fraud Detection in Financial Systems

An end-to-end machine learning project that identifies potentially fraudulent credit card transactions using imbalanced-data handling, Random Forest classification, threshold optimization, and SHAP explainability.

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent a very small proportion of total transactions.

This project develops a machine learning pipeline to:
- Explore and understand transaction data through Exploratory Data Analysis (EDA).
- Preprocess transaction data and handle class imbalance.
- Train and compare Random Forest and XGBoost classifiers.
- Optimize the classification threshold using validation data.
- Evaluate the selected model on an untouched test dataset.
- Explain individual predictions and global feature importance using SHAP.
- Provide an interactive Streamlit application for transaction analysis.

**Note:** This is an ML-based demonstration application. It is not connected to live banking systems or payment gateways.

## Technology Stack

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Imbalanced-learn, XGBoost
- **Explainable AI:** SHAP
- **Application:** Streamlit
- **Model Persistence:** Joblib

## Dataset

The project uses the publicly available Credit Card Fraud Detection dataset from Kaggle.

- Original records: 284,807
- Original features: 30
- Target: `Class`
- Legitimate transactions: 284,315
- Fraudulent transactions: 492

The dataset contains:
- `Time`
- `V1` through `V28`
- `Amount`
- `Class` (target label)

The `V1`–`V28` variables are anonymized PCA-transformed features.

The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside `data/raw/`.

## Project Architecture

```text
Real-Time-Fraud-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_fraud_model_safe.joblib
│
├── notebooks/
│
├── reports/
│   ├── eda/
│   └── models/
│
├── screenshots/
│
├── src/
│   ├── data_inspection.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── handle_imbalance.py
│   ├── leakage_safe_training.py
│   ├── threshold_optimization.py
│   └── final_evaluation.py
│
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Machine Learning Workflow

1. **Data inspection:** Examine dataset structure, missing values, duplicates and class distribution.
2. **Exploratory Data Analysis:** Study transaction amounts, time distributions and feature correlations.
3. **Preprocessing:** Remove exact duplicate records and create stratified training, validation and test sets.
4. **Feature scaling:** Fit the scaler using training data only.
5. **Class imbalance handling:** Apply SMOTE to the training data only.
6. **Model training:** Train and compare Random Forest and XGBoost.
7. **Threshold optimization:** Select the classification threshold using validation data.
8. **Final evaluation:** Evaluate the selected model on an untouched test set.
9. **Explainability:** Use SHAP to understand individual predictions and global feature contributions.
10. **Deployment demonstration:** Use Streamlit for interactive single-transaction and batch CSV analysis.

## Model Performance

The selected model is Random Forest, evaluated on an untouched stratified test set containing 42,559 transactions.

| Metric | Test Result |
|---|---:|
| Accuracy | 99.94% |
| Precision | 91.23% |
| Recall | 73.24% |
| F1-score | 81.25% |
| ROC-AUC | 95.47% |
| PR-AUC (Average Precision) | 81.21% |
| Classification Threshold | 0.60 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 42,483 | 5 |
| Actual Fraud | 19 | 52 |

The model correctly identified 52 of the 71 fraudulent transactions in the test set.

**Important:** Accuracy alone is not sufficient for evaluating this highly imbalanced dataset. Precision, recall, F1-score and PR-AUC provide additional insight into fraud detection performance.

## Explainable AI with SHAP

SHAP is integrated into the application to provide:

- Individual transaction explanations using waterfall plots.
- Global feature importance using mean absolute SHAP values.
- Feature impact visualization using beeswarm plots.

Positive SHAP contributions increase the model output for the fraud class, while negative contributions decrease it.

Since `V1`–`V28` are PCA-transformed features, their individual contributions should not be interpreted as direct merchant or customer characteristics.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Real-Time-Fraud-Detection
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the application

```powershell
python -m streamlit run app.py
```

The application will open in your browser.

The selected model file `models/best_fraud_model_safe.joblib` must be present to run the application.

## Application Features

- Single transaction fraud prediction.
- CSV-based batch transaction analysis.
- Fraud probability and risk classification.
- Downloadable prediction results.
- Individual SHAP waterfall explanations.
- Global SHAP feature importance.
- Model evaluation dashboard.
- Confusion matrix, ROC curve and precision-recall curve.

For batch analysis, provide a CSV containing the 30 input features (`Time`, `V1`–`V28`, and `Amount`). The model generates predictions; it does not independently verify the actual fraud labels in an uploaded file.

## Limitations

- This project uses a historical, publicly available dataset.
- The model has not been integrated with live banking or payment infrastructure.
- The test split is stratified and random, not chronological.
- The dataset's PCA-transformed features have limited direct interpretability.
- The model missed 19 fraudulent transactions in the final test evaluation.
- Predictions should not be treated as confirmed fraud or used as the sole basis for financial decisions.

## Future Enhancements

- Real-time transaction stream integration.
- Model monitoring and drift detection.
- Automated model retraining.
- API-based model serving.
- Enhanced fraud-risk monitoring dashboard.
- Evaluation on newer and more diverse transaction datasets.

## Author

Developed as a machine learning portfolio project focused on financial fraud detection, imbalanced classification and explainable AI.