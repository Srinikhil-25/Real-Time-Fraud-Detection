
import os
import time
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

# -----------------------------------
# PROJECT CONFIGURATION
# -----------------------------------

DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports/models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------------
# LOAD DATA
# -----------------------------------

def load_data():

    print("\nLoading datasets...")

    X_train = pd.read_csv(
        f"{DATA_DIR}/X_train_smote.csv"
    )

    y_train = pd.read_csv(
        f"{DATA_DIR}/y_train_smote.csv"
    ).squeeze("columns")

    X_test = pd.read_csv(
        f"{DATA_DIR}/X_test.csv"
    )

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    ).squeeze("columns")

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    return X_train, y_train, X_test, y_test


# -----------------------------------
# DEFINE MODELS
# -----------------------------------

def create_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4
        )
    }

    return models


# -----------------------------------
# EVALUATE MODEL
# -----------------------------------

def evaluate_model(name, model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {

        "Model": name,

        "Accuracy": accuracy_score(
            y_test, y_pred
        ),

        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test, y_prob
        ),

        "PR-AUC": average_precision_score(
            y_test, y_prob
        )
    }

    print(f"\n========== {name} ==========")

    for metric, value in results.items():

        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    return results


# -----------------------------------
# MAIN TRAINING PIPELINE
# -----------------------------------

def main():

    X_train, y_train, X_test, y_test = load_data()

    models = create_models()

    all_results = []

    print("\n========== MODEL TRAINING STARTED ==========")

    for name, model in models.items():

        print(f"\nTraining {name}...")

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        print(
            f"Training completed in {training_time:.2f} seconds"
        )

        results = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results["Training Time (seconds)"] = round(
            training_time, 2
        )

        all_results.append(results)

        model_filename = (
            name.lower().replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, model_filename)
        )

        print(f"Model saved: {model_filename}")

    # -----------------------------------
    # SAVE COMPARISON REPORT
    # -----------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        f"{REPORT_DIR}/model_comparison.csv",
        index=False
    )

    print("\n========== FINAL MODEL COMPARISON ==========")

    print(
        results_df.to_string(index=False)
    )

    print("\nModel comparison saved successfully.")

    print("\n========== TRAINING COMPLETED ==========")


if __name__ == "__main__":
    main()
