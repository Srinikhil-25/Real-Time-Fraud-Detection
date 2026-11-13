
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score
)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Real-Time Fraud Detection System")

st.markdown(
    """
    An interactive Machine Learning application for analyzing
    financial transactions, detecting potential fraud, and
    explaining model predictions using SHAP.
    """
)

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "best_fraud_model_safe.joblib"
)

TEST_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "leakage_safe"
    / "X_test.csv"
)

TEST_LABEL_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "leakage_safe"
    / "y_test.csv"
)

CLASSIFICATION_REPORT_PATH = (
    PROJECT_DIR
    / "reports"
    / "models"
    / "final_classification_report.txt"
)

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# 2. FINAL HOLDOUT TEST METRICS
# ============================================================

# These values come from the previously completed,
# untouched final holdout-test evaluation.

FINAL_METRICS = {
    "Accuracy": 0.9994,
    "Precision": 0.9123,
    "Recall": 0.7324,
    "F1-Score": 0.8125,
    "ROC-AUC": 0.9547,
    "PR-AUC": 0.8121
}

# Confusion matrix:
#                 Predicted
#                 Legit  Fraud
# Actual Legit     42483     5
# Actual Fraud        19    52

CONFUSION_MATRIX = np.array([
    [42483, 5],
    [19, 52]
])


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "scaler",
        "threshold",
        "model_name",
        "feature_columns"
    ]

    for key in required_keys:

        if key not in bundle:
            raise KeyError(
                f"Missing model bundle key: {key}"
            )

    return bundle


try:

    bundle = load_model()

    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = float(bundle["threshold"])
    model_name = bundle["model_name"]

    feature_columns = list(
        bundle["feature_columns"]
    )

except Exception as e:

    st.error(f"Unable to load model: {e}")
    st.stop()


# ============================================================
# 4. MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected Model",
        model_name
    )

with col2:
    st.metric(
        "Fraud Threshold",
        f"{threshold:.2f}"
    )

with col3:
    st.metric(
        "Input Features",
        len(feature_columns)
    )

st.divider()


# ============================================================
# 5. PREPROCESSING FUNCTION
# ============================================================

def preprocess_data(data, bundle):

    data = data.copy()

    required_features = list(
        bundle["feature_columns"]
    )

    missing_features = [
        feature
        for feature in required_features
        if feature not in data.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    data = data[required_features]

    if data.isnull().any().any():

        raise ValueError(
            "Dataset contains missing values."
        )

    data = data.apply(
        pd.to_numeric,
        errors="raise"
    )

    if not np.isfinite(
        data.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Dataset contains infinite or invalid values."
        )

    if not all(
        column in data.columns
        for column in ["Time", "Amount"]
    ):

        raise ValueError(
            "Time and Amount columns are required."
        )

    data[["Time", "Amount"]] = (
        bundle["scaler"].transform(
            data[["Time", "Amount"]]
        )
    )

    return data


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_transactions(data, bundle):

    processed_data = preprocess_data(
        data,
        bundle
    )

    model = bundle["model"]

    threshold = float(
        bundle["threshold"]
    )

    probabilities = model.predict_proba(
        processed_data
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = data.copy()

    results["Fraud_Probability"] = probabilities

    results["Prediction"] = predictions

    results["Status"] = results["Prediction"].map(
        {
            0: "Legitimate",
            1: "Fraud"
        }
    )

    def get_risk_level(probability):

        if probability >= 0.70:
            return "High Risk"

        elif probability >= 0.30:
            return "Medium Risk"

        return "Low Risk"

    results["Risk_Level"] = [
        get_risk_level(value)
        for value in probabilities
    ]

    return results


# ============================================================
# 7. INDIVIDUAL SHAP EXPLANATION
# ============================================================

def explain_transaction(data, bundle):

    model = bundle["model"]

    X = preprocess_data(
        data.iloc[[0]],
        bundle
    )

    explainer = shap.TreeExplainer(model)

    shap_result = explainer(X)

    values = np.asarray(
        shap_result.values
    )

    base_values = np.asarray(
        shap_result.base_values
    )

    if values.ndim == 3:
        values = values[:, :, 1]

    if base_values.ndim == 2:
        base_values = base_values[:, 1]

    elif base_values.ndim == 1 and len(base_values) == 2:
        base_values = base_values[1]

    explanation = shap.Explanation(
        values=values[0],
        base_values=np.asarray(
            base_values
        ).reshape(-1)[0],
        data=X.iloc[0].to_numpy(),
        feature_names=X.columns.tolist()
    )

    return explanation


# ============================================================
# 8. GLOBAL SHAP EXPLANATION
# ============================================================

def explain_global_features(bundle, sample_size=200):

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    X_test = pd.read_csv(
        TEST_DATA_PATH
    )

    required_features = list(
        bundle["feature_columns"]
    )

    X_test = X_test[required_features]

    X_sample = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=42
    ).copy()

    X_sample = preprocess_data(
        X_sample,
        bundle
    )

    explainer = shap.TreeExplainer(
        bundle["model"]
    )

    shap_result = explainer(
        X_sample
    )

    values = np.asarray(
        shap_result.values
    )

    base_values = np.asarray(
        shap_result.base_values
    )

    if values.ndim == 3:
        values = values[:, :, 1]

    if base_values.ndim == 2:
        base_values = base_values[:, 1]

    elif base_values.ndim == 1 and len(base_values) == 2:
        base_values = np.repeat(
            base_values[1],
            len(X_sample)
        )

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_sample.to_numpy(),
        feature_names=X_sample.columns.tolist()
    )

    return explanation


# ============================================================
# 9. LOAD FINAL TEST DATA FOR PERFORMANCE CURVES
# ============================================================

@st.cache_data
def load_test_data():

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test features not found: {TEST_DATA_PATH}"
        )

    if not TEST_LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Test labels not found: {TEST_LABEL_PATH}"
        )

    X_test = pd.read_csv(
        TEST_DATA_PATH
    )

    y_test = pd.read_csv(
        TEST_LABEL_PATH
    )

    if "Class" in y_test.columns:
        y_test = y_test["Class"]
    else:
        y_test = y_test.iloc[:, 0]

    return X_test, y_test.astype(int)


@st.cache_data
def calculate_test_curves(
    _model,
    _scaler,
    X_test,
    y_test,
    feature_columns
):

    X = X_test[feature_columns].copy()

    X[["Time", "Amount"]] = (
        _scaler.transform(
            X[["Time", "Amount"]]
        )
    )

    probabilities = _model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    pr_auc = average_precision_score(
    y_test,
    probabilities
)
    

    return (
        fpr,
        tpr,
        precision,
        recall,
        roc_auc,
        pr_auc
    )


# ============================================================
# 10. TRANSACTION INPUT
# ============================================================

st.header("📥 Transaction Input")

st.write(
    """
    Upload a CSV containing the 30 model features,
    or select a sample transaction from the test dataset.
    """
)

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🎲 Load Sample Transaction",
        use_container_width=True
    ):

        try:

            if not TEST_DATA_PATH.exists():
                raise FileNotFoundError(
                    "Test dataset not found."
                )

            test_data = pd.read_csv(
                TEST_DATA_PATH
            )

            sample = test_data.sample(
                n=1
            ).copy()

            st.session_state["input_data"] = sample

            st.session_state.pop(
                "prediction_results",
                None
            )

            st.session_state.pop(
                "shap_explanation",
                None
            )

            st.session_state.pop(
                "global_shap",
                None
            )

            st.success(
                "Sample transaction loaded."
            )

        except Exception as e:

            st.error(
                f"Unable to load sample: {e}"
            )


with col2:

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )

            if (
                "uploaded_filename" not in st.session_state
                or st.session_state["uploaded_filename"]
                != uploaded_file.name
            ):

                st.session_state["input_data"] = uploaded_data

                st.session_state["uploaded_filename"] = (
                    uploaded_file.name
                )

                st.session_state.pop(
                    "prediction_results",
                    None
                )

                st.session_state.pop(
                    "shap_explanation",
                    None
                )

                st.session_state.pop(
                    "global_shap",
                    None
                )

            st.success(
                f"{len(uploaded_data)} transaction(s) uploaded."
            )

        except Exception as e:

            st.error(
                f"Unable to read CSV: {e}"
            )


# ============================================================
# 11. RUN TRANSACTION PREDICTION
# ============================================================

if "input_data" in st.session_state:

    st.subheader(
        "Transaction Preview"
    )

    st.dataframe(
        st.session_state["input_data"].head(10),
        width="stretch"
    )

    if st.button(
        "🔍 Analyze Transactions",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing transactions..."
            ):

                results = predict_transactions(
                    st.session_state["input_data"],
                    bundle
                )

                st.session_state[
                    "prediction_results"
                ] = results

                st.session_state.pop(
                    "shap_explanation",
                    None
                )

                st.session_state.pop(
                    "global_shap",
                    None
                )

            st.success(
                "Transaction analysis completed!"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# 12. DISPLAY PREDICTION RESULTS
# ============================================================

if "prediction_results" in st.session_state:

    results = st.session_state[
        "prediction_results"
    ]

    st.divider()

    st.header(
        "📊 Fraud Detection Results"
    )

    total_transactions = len(results)

    fraud_count = int(
        (results["Prediction"] == 1).sum()
    )

    legitimate_count = int(
        (results["Prediction"] == 0).sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraud Detected",
        fraud_count
    )

    col3.metric(
        "Legitimate Transactions",
        legitimate_count
    )

    st.subheader(
        "Prediction Details"
    )

    display_results = results.copy()

    display_results["Fraud_Probability"] = (
        display_results["Fraud_Probability"] * 100
    ).map(
        lambda value: f"{value:.4f}%"
    )

    st.dataframe(
        display_results,
        width="stretch"
    )

    st.subheader(
        "Transaction Classification"
    )

    chart_data = pd.DataFrame(
        {
            "Transaction Type": [
                "Fraud",
                "Legitimate"
            ],
            "Count": [
                fraud_count,
                legitimate_count
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index(
            "Transaction Type"
        )
    )

    st.subheader(
        "📥 Download Prediction Report"
    )

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Prediction Results (CSV)",
        data=csv_data,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # 13. INDIVIDUAL SHAP ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🧠 Individual Transaction Explainability"
    )

    st.write(
        """
        Understand why the model assigned a particular
        fraud score to the first transaction.
        """
    )

    if st.button(
        "Explain Transaction with SHAP",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Generating SHAP explanation..."
            ):

                explanation = explain_transaction(
                    results,
                    bundle
                )

                st.session_state[
                    "shap_explanation"
                ] = explanation

        except Exception as e:

            st.error(
                f"SHAP explanation failed: {e}"
            )

    if "shap_explanation" in st.session_state:

        st.subheader(
            "Feature Contribution Analysis"
        )

        st.write(
            """
            Features pushing the model output higher increase
            the fraud-class score. Features pushing it lower
            decrease that score.
            """
        )

        try:

            shap.plots.waterfall(
                st.session_state["shap_explanation"],
                max_display=12,
                show=False
            )

            fig = plt.gcf()

            st.pyplot(
                fig,
                width="stretch"
            )

            plt.close(fig)

        except Exception as e:

            st.error(
                f"Unable to display SHAP chart: {e}"
            )


    # ========================================================
    # 14. GLOBAL SHAP ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🌍 Global Feature Importance"
    )

    st.write(
        """
        Understand which transaction features influence
        fraud predictions most across multiple transactions.
        """
    )

    st.caption(
        "Uses up to 200 transactions from the untouched test dataset."
    )

    if st.button(
        "Generate Global SHAP Analysis",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Calculating global SHAP values..."
            ):

                global_explanation = explain_global_features(
                    bundle,
                    sample_size=200
                )

                st.session_state[
                    "global_shap"
                ] = global_explanation

            st.success(
                "Global SHAP analysis completed successfully!"
            )

        except Exception as e:

            st.error(
                f"Global SHAP analysis failed: {e}"
            )

    if "global_shap" in st.session_state:

        global_explanation = st.session_state[
            "global_shap"
        ]

        st.subheader(
            "1. Overall Feature Importance"
        )

        st.write(
            "Features are ranked by mean absolute SHAP value."
        )

        try:

            shap.plots.bar(
                global_explanation,
                max_display=15,
                show=False
            )

            fig = plt.gcf()

            st.pyplot(
                fig,
                width="stretch"
            )

            plt.close(fig)

        except Exception as e:

            st.error(
                f"Unable to display bar plot: {e}"
            )

        st.subheader(
            "2. Feature Impact Distribution"
        )

        st.write(
            """
            Each point represents a transaction.
            Positive SHAP values push the fraud-class output higher;
            negative values push it lower.
            """
        )

        try:

            shap.plots.beeswarm(
                global_explanation,
                max_display=15,
                show=False
            )

            fig = plt.gcf()

            st.pyplot(
                fig,
                width="stretch"
            )

            plt.close(fig)

        except Exception as e:

            st.error(
                f"Unable to display beeswarm plot: {e}"
            )


# ============================================================
# 15. MODEL PERFORMANCE DASHBOARD
# ============================================================

st.divider()

st.header(
    "📈 Model Performance Dashboard"
)

st.markdown(
    """
    Evaluation results from the final untouched holdout-test
    dataset. These results were not used for model training
    or threshold selection.
    """
)

st.caption(
    "Final test set: 42,559 transactions, including 71 fraud cases."
)


# ------------------------------------------------------------
# PERFORMANCE METRIC CARDS
# ------------------------------------------------------------

st.subheader(
    "1. Evaluation Metrics"
)

metric_names = list(
    FINAL_METRICS.keys()
)

metric_columns = st.columns(3)

for index, metric_name in enumerate(metric_names):

    with metric_columns[index % 3]:

        st.metric(
            metric_name,
            f"{FINAL_METRICS[metric_name] * 100:.2f}%"
        )


st.caption(
    """
    Accuracy measures overall correctness. Precision measures
    how many flagged transactions were actually fraud. Recall
    measures how many actual fraud cases were detected.
    """
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

st.divider()

st.subheader(
    "2. Confusion Matrix"
)

st.write(
    """
    The confusion matrix compares actual transaction labels
    with the model's predicted labels.
    """
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

image = ax.imshow(
    CONFUSION_MATRIX,
    interpolation="nearest",
    cmap="Blues"
)

ax.set_title(
    "Final Test Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Legitimate",
    "Fraud"
])

ax.set_yticklabels([
    "Legitimate",
    "Fraud"
])

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            f"{CONFUSION_MATRIX[i, j]:,}",
            ha="center",
            va="center",
            fontsize=13,
            color="black"
        )

fig.colorbar(
    image,
    ax=ax
)

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ------------------------------------------------------------
# CONFUSION MATRIX INTERPRETATION
# ------------------------------------------------------------

tn, fp, fn, tp = CONFUSION_MATRIX.ravel()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "True Negatives",
    f"{tn:,}"
)

col2.metric(
    "False Positives",
    f"{fp:,}"
)

col3.metric(
    "False Negatives",
    f"{fn:,}"
)

col4.metric(
    "True Positives",
    f"{tp:,}"
)

st.info(
    f"""
    The model correctly detected {tp} out of 71 fraud cases.
    It missed {fn} fraud cases and generated {fp} false alerts.
    """
)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

st.divider()

st.subheader(
    "3. Classification Report"
)

if CLASSIFICATION_REPORT_PATH.exists():

    report_text = CLASSIFICATION_REPORT_PATH.read_text(
        encoding="utf-8"
    )

    st.code(
        report_text,
        language="text"
    )

else:

    st.warning(
        "Saved classification report was not found. "
        "The confusion matrix and evaluation metrics are "
        "still displayed above."
    )


# ------------------------------------------------------------
# ROC AND PRECISION-RECALL CURVES
# ------------------------------------------------------------

st.divider()

st.subheader(
    "4. ROC Curve and Precision-Recall Curve"
)

st.write(
    """
    These curves are calculated using prediction probabilities
    from the saved final holdout-test dataset.
    """
)

if st.button(
    "Generate ROC and Precision-Recall Curves",
    use_container_width=True
):

    try:

        with st.spinner(
            "Calculating evaluation curves..."
        ):

            X_test, y_test = load_test_data()

            (
                fpr,
                tpr,
                precision_values,
                recall_values,
                roc_auc_value,
                pr_auc_value
            ) = calculate_test_curves(
                model,
                scaler,
                X_test,
                y_test,
                feature_columns
            )

            st.session_state[
                "performance_curves"
            ] = {
                "fpr": fpr,
                "tpr": tpr,
                "precision": precision_values,
                "recall": recall_values,
                "roc_auc": roc_auc_value,
                "pr_auc": pr_auc_value
            }

        st.success(
            "Evaluation curves generated successfully!"
        )

    except Exception as e:

        st.error(
            f"Unable to generate evaluation curves: {e}"
        )


if "performance_curves" in st.session_state:

    curves = st.session_state[
        "performance_curves"
    ]

    st.write(
        f"Calculated ROC-AUC: **{curves['roc_auc']:.4f}**"
    )

    st.write(
        f"Calculated PR-AUC: **{curves['pr_auc']:.4f}**"
    )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.markdown(
        "### Receiver Operating Characteristic (ROC)"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        curves["fpr"],
        curves["tpr"],
        label=f"ROC-AUC = {curves['roc_auc']:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    st.markdown(
        "### Precision-Recall Curve"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        curves["recall"],
        curves["precision"],
        label=f"PR-AUC = {curves['pr_auc']:.4f}"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall Curve"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Real-Time Fraud Detection System | Machine Learning Project

    Built using Python, Scikit-learn, Random Forest,
    Streamlit and SHAP.

    This is an ML-based demonstration application. It is not
    connected to live banking systems and should not be used
    as the sole basis for real financial decisions.
    """
)
