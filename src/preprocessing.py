
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

DATA_PATH = "data/raw/creditcard.csv"
OUTPUT_DIR = "data/processed"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

SCALE_COLUMNS = ["Time", "Amount"]


# ==========================================
# LOAD DATASET
# ==========================================

def load_data():

    print("\nLoading original dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    return df


# ==========================================
# DATA CLEANING
# ==========================================

def clean_data(df):

    print("\n========== DATA CLEANING ==========")

    missing_values = df.isnull().sum().sum()

    print("Missing values:", missing_values)

    duplicates = df.duplicated().sum()

    print("Exact duplicate rows:", duplicates)

    # Remove exact duplicate rows
    df = df.drop_duplicates(keep="first").copy()

    print("Cleaned dataset shape:", df.shape)

    print("\nClass distribution after cleaning:")

    print(df["Class"].value_counts())

    return df


# ==========================================
# FEATURE AND TARGET SEPARATION
# ==========================================

def separate_features(df):

    X = df.drop(columns=["Class"])

    y = df["Class"]

    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

def split_data(X, y):

    print("\n========== TRAIN TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining class distribution:")

    print(y_train.value_counts())

    print("\nTesting class distribution:")

    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# ==========================================
# FEATURE SCALING
# ==========================================

def scale_features(X_train, X_test):

    print("\n========== FEATURE SCALING ==========")

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data to prevent leakage
    X_train[SCALE_COLUMNS] = scaler.fit_transform(
        X_train[SCALE_COLUMNS]
    )

    # Apply the same training scaler to test data
    X_test[SCALE_COLUMNS] = scaler.transform(
        X_test[SCALE_COLUMNS]
    )

    print("Scaled columns:", SCALE_COLUMNS)

    return X_train, X_test, scaler


# ==========================================
# SAVE PROCESSED DATA
# ==========================================

def save_data(X_train, X_test, y_train, y_test, scaler):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train.to_csv(
        f"{OUTPUT_DIR}/X_train.csv",
        index=False
    )

    X_test.to_csv(
        f"{OUTPUT_DIR}/X_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{OUTPUT_DIR}/y_train.csv",
        index=False,
        header=True
    )

    y_test.to_csv(
        f"{OUTPUT_DIR}/y_test.csv",
        index=False,
        header=True
    )

    joblib.dump(
        scaler,
        f"{MODEL_DIR}/feature_scaler.joblib"
    )

    print("\nProcessed data saved successfully!")

    print("Location:", OUTPUT_DIR)
    print("Scaler saved in:", MODEL_DIR)


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    df = load_data()

    df = clean_data(df)

    X, y = separate_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )

    print("\n===================================")
    print("PREPROCESSING COMPLETED!")
    print("===================================")


if __name__ == "__main__":
    main()
