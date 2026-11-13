
import pandas as pd
import numpy as np

# Dataset location
DATA_PATH = "data/raw/creditcard.csv"


def load_dataset():
    print("\nLoading credit card dataset...")

    try:
        df = pd.read_csv(DATA_PATH)

        print("\nDataset loaded successfully!")

        return df

    except FileNotFoundError:
        print("\nERROR: Dataset not found!")
        print("Place creditcard.csv inside data/raw/")
        return None


def inspect_dataset(df):
    print("\n========== DATASET OVERVIEW ==========")

    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\n========== FRAUD DISTRIBUTION ==========")

    print(df["Class"].value_counts())

    print("\nFraud percentages:")
    print(df["Class"].value_counts(normalize=True) * 100)

    print("\n========== STATISTICAL SUMMARY ==========")

    print(df.describe())


if __name__ == "__main__":

    dataset = load_dataset()

    if dataset is not None:
        inspect_dataset(dataset)