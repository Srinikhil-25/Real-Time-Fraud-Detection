
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)

# -----------------------------------
# CONFIGURATION
# -----------------------------------

DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports/models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42


# -----------------------------------
# LOAD ORIGINAL TRAINING DATA
# -----------------------------------

def load_data():

    print("\nLoading original training data...")

    X = pd.read_csv(f"{DATA_DIR}/X_train.csv")

    y = pd.read_csv(
        f"{DATA_DIR}/y_train.csv"
    ).squeeze("columns")

    print("Original training shape:", X.shape)

    return X, y


# -----------------------------------
# CREATE VALIDATION SPLIT
# -----------------------------------

def create_validation_split(X, y):

    print("\nCreating stratified validation split...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nValidation class distribution:")
    print(y_val.value_counts())

    return X_train, X_val, y_train, y_val


# -----------------------------------
# APPLY SMOTE TO TRAINING DATA ONLY
# -----------------------------------

def balance_training_data(X_train, y_train):

    print("\nApplying SMOTE to training data...")

    smote = SMOTE(
        sampling_strategy=0.5,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(y_resampled.value_counts())

    return X_resampled, y_resampled


# -----------------------------------
# CREATE MODELS
# -----------------------------------

def create_models():

    return {

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": __import__("xgboost").XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4
        )
    }


# -----------------------------------
# EVALUATE THRESHOLDS
# -----------------------------------

def evaluate_thresholds(name, model, X_val, y_val):

    probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        probabilities
    )

    # Use thresholds from 0.05 to 0.95.
    selected_thresholds = np.arange(0.05, 0.96, 0.05)

    results = []

    print(f"\n========== {name} THRESHOLD RESULTS ==========")

    for threshold in selected_thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        p = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        r = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Threshold": round(float(threshold), 2),
            "Precision": p,
            "Recall": r,
            "F1 Score": f1
        })

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {p:.4f} | "
            f"Recall: {r:.4f} | "
            f"F1: {f1:.4f}"
        )

    return results, probabilities


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def main():

    X, y = load_data()

    X_train, X_val, y_train, y_val = create_validation_split(
        X, y
    )

    X_train_smote, y_train_smote = balance_training_data(
        X_train,
        y_train
    )

    models = create_models()

    all_results = []

    best_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        start_time = time.time()

        model.fit(
            X_train_smote,
            y_train_smote
        )

        print(
            f"Training time: {time.time() - start_time:.2f} seconds"
        )

        results, probabilities = evaluate_thresholds(
            name,
            model,
            X_val,
            y_val
        )

        all_results.extend(results)

        best_index = max(
            range(len(results)),
            key=lambda i: results[i]["F1 Score"]
        )

        best_result = results[best_index]

        best_models[name] = {
            "model": model,
            "threshold": best_result["Threshold"],
            "precision": best_result["Precision"],
            "recall": best_result["Recall"],
            "f1": best_result["F1 Score"],
            "pr_auc": average_precision_score(
                y_val,
                probabilities
            )
        }

        print(f"\nBest threshold for {name}:")
        print(best_result)

        filename = (
            name.lower().replace(" ", "_")
            + "_validated.joblib"
        )

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, filename)
        )

    # -----------------------------------
    # SAVE THRESHOLD RESULTS
    # -----------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        f"{REPORT_DIR}/threshold_comparison.csv",
        index=False
    )

    # -----------------------------------
    # SELECT BEST MODEL
    # -----------------------------------

    best_name = max(
        best_models,
        key=lambda name: best_models[name]["f1"]
    )

    best = best_models[best_name]

    joblib.dump(
        {
            "model": best["model"],
            "threshold": best["threshold"]
        },
        f"{MODEL_DIR}/best_fraud_model.joblib"
    )

    print("\n========== BEST VALIDATION MODEL ==========")

    print("Model:", best_name)
    print("Threshold:", best["threshold"])
    print("Precision:", round(best["precision"], 4))
    print("Recall:", round(best["recall"], 4))
    print("F1 Score:", round(best["f1"], 4))
    print("PR-AUC:", round(best["pr_auc"], 4))

    print("\nThreshold results saved successfully.")
    print("\nVALIDATION PIPELINE COMPLETED!")


if __name__ == "__main__":
    main()
