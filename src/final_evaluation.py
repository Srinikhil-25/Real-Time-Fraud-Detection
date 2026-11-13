
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "models/best_fraud_model_safe.joblib"

DATA_DIR = "data/processed/leakage_safe"

REPORT_DIR = "reports/models"

os.makedirs(REPORT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD MODEL AND TEST DATA
# ==========================================

def load_artifacts():

    print("\nLoading selected fraud detection model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    model_name = bundle["model_name"]
    feature_columns = bundle["feature_columns"]

    print("Selected model:", model_name)
    print("Selected threshold:", threshold)

    print("\nLoading fresh final test data...")

    X_test = pd.read_csv(
        f"{DATA_DIR}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    )["Class"]

    return (
        model,
        scaler,
        threshold,
        model_name,
        feature_columns,
        X_test,
        y_test
    )


# ==========================================
# 3. PREPROCESS TEST DATA
# ==========================================

def preprocess_test_data(
    X_test,
    scaler,
    feature_columns
):

    print("\nPreprocessing final test data...")

    # Preserve the original feature order
    X_test = X_test[feature_columns].copy()

    # Transform only the scaled columns
    scale_columns = [
        "Time",
        "Amount"
    ]

    X_test[scale_columns] = scaler.transform(
        X_test[scale_columns]
    )

    print("Test preprocessing completed.")

    return X_test


# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================

def generate_predictions(
    model,
    X_test,
    threshold
):

    print("\nGenerating fraud predictions...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return probabilities, predictions


# ==========================================
# 5. CALCULATE FINAL METRICS
# ==========================================

def evaluate_model(
    y_test,
    predictions,
    probabilities
):

    print("\nCalculating final evaluation metrics...")

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "PR-AUC": average_precision_score(
            y_test,
            probabilities
        )

    }

    return metrics


# ==========================================
# 6. SAVE CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Legitimate",
            "Fraud"
        ],
        yticklabels=[
            "Legitimate",
            "Fraud"
        ]
    )

    plt.title(
        "Final Test Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    output_path = os.path.join(
        REPORT_DIR,
        "final_test_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        "\nConfusion matrix saved:",
        output_path
    )

    return cm


# ==========================================
# 7. SAVE EVALUATION REPORT
# ==========================================

def save_reports(
    metrics,
    y_test,
    predictions,
    probabilities,
    model_name,
    threshold
):

    # Save numerical metrics
    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.insert(
        0,
        "Model",
        model_name
    )

    metrics_df["Threshold"] = threshold

    metrics_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "final_test_metrics.csv"
        ),
        index=False
    )

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )

    with open(
        os.path.join(
            REPORT_DIR,
            "final_classification_report.txt"
        ),
        "w"
    ) as file:

        file.write(report)

    # Save individual predictions
    predictions_df = pd.DataFrame({

        "Actual_Class": y_test.to_numpy(),

        "Predicted_Class": predictions,

        "Fraud_Probability": probabilities

    })

    predictions_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "final_test_predictions.csv"
        ),
        index=False
    )

    print("\nEvaluation reports saved.")


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

def main():

    (
        model,
        scaler,
        threshold,
        model_name,
        feature_columns,
        X_test,
        y_test
    ) = load_artifacts()

    X_test = preprocess_test_data(
        X_test,
        scaler,
        feature_columns
    )

    probabilities, predictions = generate_predictions(
        model,
        X_test,
        threshold
    )

    metrics = evaluate_model(
        y_test,
        predictions,
        probabilities
    )

    cm = save_confusion_matrix(
        y_test,
        predictions
    )

    save_reports(
        metrics,
        y_test,
        predictions,
        probabilities,
        model_name,
        threshold
    )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\n" + "=" * 50)
    print("FINAL FRAUD DETECTION EVALUATION")
    print("=" * 50)

    print("Selected model:", model_name)
    print("Classification threshold:", threshold)

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    print("\nTrue Negatives:", cm[0, 0])
    print("False Positives:", cm[0, 1])
    print("False Negatives:", cm[1, 0])
    print("True Positives:", cm[1, 1])

    print("\nFinal evaluation completed successfully!")


if __name__ == "__main__":

    main()
