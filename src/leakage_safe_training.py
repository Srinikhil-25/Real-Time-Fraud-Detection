
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = "data/raw"
MODEL_DIR = "models"
REPORT_DIR = "reports/models"
SAFE_DATA_DIR = "data/processed/leakage_safe"

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SAFE_DATA_DIR, exist_ok=True)


# ==========================================
# 2. LOAD ORIGINAL DATASET
# ==========================================

def load_data():

    print("\nLoading original Kaggle dataset...")

    file_path = os.path.join(
        DATA_DIR,
        "creditcard.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("Original dataset shape:", df.shape)

    # Remove exact duplicate records
    duplicate_count = df.duplicated().sum()

    print("Duplicate records:", duplicate_count)

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    print("Cleaned dataset shape:", df.shape)

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nFraud percentage:")
    print(round(y.mean() * 100, 4), "%")

    return X, y


# ==========================================
# 3. CREATE TRAIN / VALIDATION / TEST SPLITS
# ==========================================

def split_data(X, y):

    print("\nCreating fresh stratified splits...")

    # First split: 85% development, 15% final test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split: 70% training, 15% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=(0.15 / 0.85),
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print("\nDataset split results:")

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Final test:", X_test.shape)

    print("\nFraud distribution:")

    print(
        "Training:",
        y_train.value_counts().to_dict()
    )

    print(
        "Validation:",
        y_val.value_counts().to_dict()
    )

    print(
        "Final test:",
        y_test.value_counts().to_dict()
    )

    # Save the raw final test data.
    # It will only be used after model selection.

    X_test.to_csv(
        f"{SAFE_DATA_DIR}/X_test.csv",
        index=False
    )

    y_test.to_frame(name="Class").to_csv(
        f"{SAFE_DATA_DIR}/y_test.csv",
        index=False
    )

    print("\nRaw final test data saved.")

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        X_test,
        y_test
    )


# ==========================================
# 4. SCALE FEATURES WITHOUT DATA LEAKAGE
# ==========================================

def scale_data(X_train, X_val, X_test):

    print("\nScaling numerical features...")

    scaler = StandardScaler()

    # Fit scaler only on training data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    scale_columns = [
        "Time",
        "Amount"
    ]

    X_train_scaled[scale_columns] = scaler.fit_transform(
        X_train[scale_columns]
    )

    # Transform validation and test data only
    X_val_scaled[scale_columns] = scaler.transform(
        X_val[scale_columns]
    )

    X_test_scaled[scale_columns] = scaler.transform(
        X_test[scale_columns]
    )

    joblib.dump(
        scaler,
        f"{MODEL_DIR}/leakage_safe_scaler.joblib"
    )

    print("Scaling completed.")
    print("Scaler fitted only on training data.")

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# ==========================================
# 5. HANDLE CLASS IMBALANCE USING SMOTE
# ==========================================

def balance_data(X_train, y_train):

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

    print("\nClass distribution after SMOTE:")

    print(
        pd.Series(y_resampled).value_counts()
    )

    print("\nSMOTE completed.")

    return X_resampled, y_resampled


# ==========================================
# 6. DEFINE MACHINE LEARNING MODELS
# ==========================================

def create_models():

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_split=5,
            random_state=RANDOM_STATE,
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
            random_state=RANDOM_STATE,
            n_jobs=4
        )
    }

    return models


# ==========================================
# 7. FIND BEST CLASSIFICATION THRESHOLD
# ==========================================

def find_best_threshold(
    model,
    X_val,
    y_val,
    model_name
):

    print(
        f"\nFinding optimal threshold for {model_name}..."
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    results = []

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
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

            "Model": model_name,

            "Threshold": round(
                float(threshold), 2
            ),

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1

        })

    results_df = pd.DataFrame(results)

    # Select threshold with highest validation F1
    best = results_df.loc[
        results_df["F1 Score"].idxmax()
    ]

    # Calculate validation PR-AUC
    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    return best, results_df, pr_auc


# ==========================================
# 8. MAIN TRAINING PIPELINE
# ==========================================

def main():

    # Load raw data
    X, y = load_data()

    # Create fresh splits
    (
        X_train,
        X_val,
        y_train,
        y_val,
        X_test,
        y_test
    ) = split_data(X, y)

    # Scale without leakage
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_data(
        X_train,
        X_val,
        X_test
    )

    # Apply SMOTE only to training data
    X_train_smote, y_train_smote = balance_data(
        X_train_scaled,
        y_train
    )

    # Define models
    models = create_models()

    all_results = []
    best_models = {}

    # Train and compare models
    for name, model in models.items():

        print("\n" + "=" * 45)
        print("Training:", name)
        print("=" * 45)

        start_time = time.time()

        model.fit(
            X_train_smote,
            y_train_smote
        )

        training_time = time.time() - start_time

        print(
            "Training time:",
            round(training_time, 2),
            "seconds"
        )

        # Optimize threshold using validation data
        best, results_df, pr_auc = find_best_threshold(
            model,
            X_val_scaled,
            y_val,
            name
        )

        all_results.append(results_df)

        print("\nBest validation result:")

        print(
            best.to_dict()
        )

        print(
            "Validation PR-AUC:",
            round(pr_auc, 4)
        )

        best_models[name] = {

            "model": model,

            "threshold": float(
                best["Threshold"]
            ),

            "precision": float(
                best["Precision"]
            ),

            "recall": float(
                best["Recall"]
            ),

            "f1": float(
                best["F1 Score"]
            ),

            "pr_auc": float(pr_auc),

            "training_time": training_time

        }

    # ======================================
    # 9. SELECT BEST MODEL
    # ======================================

    best_name = max(
        best_models,
        key=lambda name: best_models[name]["f1"]
    )

    best = best_models[best_name]

    final_bundle = {

        "model": best["model"],

        "scaler": scaler,

        "threshold": best["threshold"],

        "model_name": best_name,

        "feature_columns": list(X.columns)

    }

    joblib.dump(
        final_bundle,
        f"{MODEL_DIR}/best_fraud_model_safe.joblib"
    )

    # ======================================
    # 10. SAVE RESULTS
    # ======================================

    comparison = pd.concat(
        all_results,
        ignore_index=True
    )

    comparison.to_csv(
        f"{REPORT_DIR}/leakage_safe_thresholds.csv",
        index=False
    )

    # Save model summary
    summary = pd.DataFrame([
        {
            "Model": name,
            "Threshold": values["threshold"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1 Score": values["f1"],
            "PR-AUC": values["pr_auc"],
            "Training Time": values["training_time"]
        }
        for name, values in best_models.items()
    ])

    summary.to_csv(
        f"{REPORT_DIR}/leakage_safe_model_comparison.csv",
        index=False
    )

    # ======================================
    # 11. DISPLAY FINAL VALIDATION RESULTS
    # ======================================

    print("\n" + "=" * 50)
    print("SELECTED MODEL")
    print("=" * 50)

    print("Model:", best_name)

    print(
        "Threshold:",
        best["threshold"]
    )

    print(
        "Precision:",
        round(best["precision"], 4)
    )

    print(
        "Recall:",
        round(best["recall"], 4)
    )

    print(
        "F1 Score:",
        round(best["f1"], 4)
    )

    print(
        "PR-AUC:",
        round(best["pr_auc"], 4)
    )

    print("\nSaved files:")

    print(
        f"{MODEL_DIR}/best_fraud_model_safe.joblib"
    )

    print(
        f"{REPORT_DIR}/leakage_safe_thresholds.csv"
    )

    print(
        f"{REPORT_DIR}/leakage_safe_model_comparison.csv"
    )

    print(
        f"{SAFE_DATA_DIR}/X_test.csv"
    )

    print(
        f"{SAFE_DATA_DIR}/y_test.csv"
    )

    print("\nLeakage-safe training completed successfully!")


# ==========================================
# 12. EXECUTION
# ==========================================

if __name__ == "__main__":

    main()
