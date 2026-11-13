
import os
import pandas as pd

from imblearn.over_sampling import SMOTE


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "data/processed"

RANDOM_STATE = 42


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

def load_data():

    print("\nLoading processed datasets...")

    X_train = pd.read_csv(
        f"{DATA_DIR}/X_train.csv"
    )

    y_train = pd.read_csv(
        f"{DATA_DIR}/y_train.csv"
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


# ==========================================
# ORIGINAL CLASS DISTRIBUTION
# ==========================================

def show_distribution(y):

    print("\nClass distribution:")

    print(y.value_counts().sort_index())

    print("\nClass percentages:")

    print(
        (y.value_counts(normalize=True) * 100)
        .sort_index()
    )


# ==========================================
# APPLY SMOTE
# ==========================================

def apply_smote(X_train, y_train):

    print("\n========== APPLYING SMOTE ==========")

    print("\nBefore SMOTE:")

    show_distribution(y_train)

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

    show_distribution(y_resampled)

    print("\nResampled feature shape:", X_resampled.shape)

    return X_resampled, y_resampled


# ==========================================
# SAVE RESAMPLED DATA
# ==========================================

def save_resampled_data(X_resampled, y_resampled):

    os.makedirs(DATA_DIR, exist_ok=True)

    X_resampled.to_csv(
        f"{DATA_DIR}/X_train_smote.csv",
        index=False
    )

    y_resampled.to_csv(
        f"{DATA_DIR}/y_train_smote.csv",
        index=False,
        header=True
    )

    print("\nResampled training data saved successfully!")


# ==========================================
# MAIN
# ==========================================

def main():

    X_train, y_train, X_test, y_test = load_data()

    X_resampled, y_resampled = apply_smote(
        X_train,
        y_train
    )

    save_resampled_data(
        X_resampled,
        y_resampled
    )

    print("\n===================================")
    print("SMOTE PROCESS COMPLETED!")
    print("Test dataset remains unchanged.")
    print("===================================")


if __name__ == "__main__":
    main()
