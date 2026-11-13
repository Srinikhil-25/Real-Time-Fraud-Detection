
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# PROJECT CONFIGURATION
# ----------------------------------

DATA_PATH = "data/raw/creditcard.csv"
OUTPUT_DIR = "reports/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ----------------------------------
# LOAD DATASET
# ----------------------------------

def load_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    return df


# ----------------------------------
# DUPLICATE ANALYSIS
# ----------------------------------

def analyze_duplicates(df):

    print("\n========== DUPLICATE ANALYSIS ==========")

    duplicate_mask = df.duplicated(keep=False)

    duplicate_rows = df[duplicate_mask]

    print("Duplicate rows (including first occurrences):",
          len(duplicate_rows))

    print("Duplicate records beyond first occurrences:",
          df.duplicated().sum())

    print("\nClass distribution among duplicate rows:")

    print(duplicate_rows["Class"].value_counts())

    print("\nClass distribution among duplicate rows (%):")

    print(duplicate_rows["Class"].value_counts(normalize=True) * 100)


# ----------------------------------
# CLASS DISTRIBUTION
# ----------------------------------


def plot_class_distribution(df):

    counts = df["Class"].value_counts().sort_index()

    plt.figure(figsize=(9, 6))

    ax = sns.barplot(
        x=["Legitimate", "Fraud"],
        y=[counts[0], counts[1]],
        hue=["Legitimate", "Fraud"],
        palette=["#16A34A", "#DC6262"],
        legend=False
    )

    ax.set_yscale("log")

    for i, count in enumerate([counts[0], counts[1]]):
        ax.text(
            i,
            count * 1.15,
            f"{count:,}",
            ha="center",
            fontweight="bold"
        )

    plt.title(
        "Legitimate vs Fraudulent Transactions",
        fontsize=15
    )

    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions (Log Scale)")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/class_distribution.png",
        dpi=300
    )

    plt.close()


def investigate_duplicate_labels(df):

    print("\n========== DUPLICATE LABEL INVESTIGATION ==========")

    feature_columns = df.columns.drop("Class").tolist()

    duplicate_features = df[
        df.duplicated(
            subset=feature_columns,
            keep=False
        )
    ]

    if duplicate_features.empty:
        print("No duplicate feature records found.")
        return

    duplicate_groups = duplicate_features.groupby(
        feature_columns,
        dropna=False
    )["Class"].agg(["nunique", "size"])

    conflicting_groups = duplicate_groups[
        duplicate_groups["nunique"] > 1
    ]

    print(
        "Rows with repeated feature combinations:",
        len(duplicate_features)
    )

    print(
        "Repeated feature groups:",
        len(duplicate_groups)
    )

    print(
        "Groups with conflicting class labels:",
        len(conflicting_groups)
    )

    if not conflicting_groups.empty:
        print("\nConflicting groups:")
        print(conflicting_groups.head(10))

    else:
        print(
            "\nNo conflicting class labels found "
            "among identical feature combinations."
        )

# ----------------------------------
# TRANSACTION AMOUNT ANALYSIS
# ----------------------------------

def plot_transaction_amount(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Amount",
        bins=100,
        color="steelblue"
    )

    plt.xscale("symlog")

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/amount_distribution.png",
        dpi=300
    )

    plt.close()


# ----------------------------------
# FRAUD VS LEGITIMATE AMOUNTS
# ----------------------------------

def plot_amount_comparison(df):

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount",
        showfliers=False
    )

    plt.yscale("symlog")

    plt.xticks([0, 1], ["Legitimate", "Fraud"])

    plt.title("Transaction Amount Comparison")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/amount_comparison.png",
        dpi=300
    )

    plt.close()


# ----------------------------------
# TRANSACTION TIME ANALYSIS
# ----------------------------------

def plot_transaction_time(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=100,
        element="step",
        stat="count",
        common_norm=False
    )

    plt.title("Transaction Time Distribution")
    plt.xlabel("Elapsed Time (Seconds)")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/time_distribution.png",
        dpi=300
    )

    plt.close()


# ----------------------------------
# CORRELATION ANALYSIS
# ----------------------------------

def plot_correlation(df):

    correlations = df.corr(numeric_only=True)["Class"]

    correlations = correlations.drop("Class")

    top_features = correlations.abs().nlargest(12).index

    selected_features = list(top_features) + ["Class"]

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        df[selected_features].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Top Feature Correlations with Fraud Class")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    print("\nTop features correlated with fraud:")

    print(correlations.loc[top_features].sort_values(
        key=abs,
        ascending=False
    ))


# ----------------------------------
# FRAUD STATISTICS
# ----------------------------------

def fraud_statistics(df):

    print("\n========== FRAUD STATISTICS ==========")

    print("\nAverage transaction amount:")

    print(df.groupby("Class")["Amount"].mean())

    print("\nMedian transaction amount:")

    print(df.groupby("Class")["Amount"].median())

    print("\nTransaction count:")

    print(df["Class"].value_counts())


# ----------------------------------
# MAIN EXECUTION
# ----------------------------------

if __name__ == "__main__":

    df = load_data()

    analyze_duplicates(df)

    investigate_duplicate_labels(df)

    fraud_statistics(df)

    plot_class_distribution(df)

    plot_transaction_amount(df)

    plot_amount_comparison(df)

    plot_transaction_time(df)

    plot_correlation(df)

    print("\n===================================")
    print("EDA COMPLETED SUCCESSFULLY!")
    print("Visualizations saved in:", OUTPUT_DIR)
    print("===================================")
