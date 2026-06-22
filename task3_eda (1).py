
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------
DATA_PATH = r"C:\Users\ADMIN\Downloads\customer_sales.csv"          # <-- change this to your dataset path
OUTPUT_DIR = "eda_outputs"      # folder where plots will be saved
sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    """Load dataset and handle common errors."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def basic_info(df):
    """Print structural overview of the dataset."""
    print("=" * 60)
    print("1. BASIC INFORMATION")
    print("=" * 60)
    print("\nShape:", df.shape)
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
    print("\nLast 5 rows:\n", df.tail())
    print("\nInfo:")
    df.info()


def missing_values(df):
    """Check for missing values."""
    print("\n" + "=" * 60)
    print("2. MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )
    if missing_df.empty:
        print("No missing values found.")
    else:
        print(missing_df)

    # Visualize missing values
    if missing_df.shape[0] > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/missing_values_heatmap.png")
        plt.close()


def duplicate_check(df):
    """Check for duplicate rows."""
    print("\n" + "=" * 60)
    print("3. DUPLICATE ROWS")
    print("=" * 60)
    dup_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {dup_count}")


def summary_statistics(df):
    """Print descriptive statistics for numeric and categorical columns."""
    print("\n" + "=" * 60)
    print("4. SUMMARY STATISTICS")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()

    if numeric_cols:
        print("\nNumeric columns summary:\n", df[numeric_cols].describe().T)
    if categorical_cols:
        print("\nCategorical columns summary:\n", df[categorical_cols].describe().T)

    return numeric_cols, categorical_cols


def univariate_analysis(df, numeric_cols, categorical_cols):
    """Generate distribution plots for numeric and categorical columns."""
    print("\n" + "=" * 60)
    print("5. UNIVARIATE ANALYSIS (saved as images)")
    print("=" * 60)

    # Numeric distributions
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dist_{col}.png")
        plt.close()

    # Boxplots for outlier detection
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color="lightcoral")
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/box_{col}.png")
        plt.close()

    # Categorical value counts
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        top_categories = df[col].value_counts().head(15)
        sns.barplot(x=top_categories.values, y=top_categories.index,
                    hue=top_categories.index, palette="viridis", legend=False)
        plt.title(f"Top categories in {col}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cat_{col}.png")
        plt.close()

    print(f"Plots saved in '{OUTPUT_DIR}/' folder.")


def correlation_analysis(df, numeric_cols):
    """Generate correlation heatmap for numeric features."""
    print("\n" + "=" * 60)
    print("6. CORRELATION ANALYSIS")
    print("=" * 60)

    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    corr = df[numeric_cols].corr()
    print(corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()


def outlier_detection(df, numeric_cols):
    """Detect outliers using the IQR method."""
    print("\n" + "=" * 60)
    print("7. OUTLIER DETECTION (IQR Method)")
    print("=" * 60)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers (bounds: {lower:.2f} to {upper:.2f})")


def bivariate_analysis(df, numeric_cols, categorical_cols):
    """Generate pairwise scatter plots for numeric columns (sampled if too many)."""
    print("\n" + "=" * 60)
    print("8. BIVARIATE ANALYSIS")
    print("=" * 60)

    if len(numeric_cols) >= 2:
        sample_df = df[numeric_cols].sample(min(500, len(df)), random_state=42)
        sns.pairplot(sample_df)
        plt.savefig(f"{OUTPUT_DIR}/pairplot.png")
        plt.close()
        print("Pairplot saved.")
    else:
        print("Not enough numeric columns for pairplot.")


def main():
    df = load_data(DATA_PATH)
    basic_info(df)
    missing_values(df)
    duplicate_check(df)
    numeric_cols, categorical_cols = summary_statistics(df)
    univariate_analysis(df, numeric_cols, categorical_cols)
    correlation_analysis(df, numeric_cols)
    outlier_detection(df, numeric_cols)
    bivariate_analysis(df, numeric_cols, categorical_cols)

    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE — check the 'eda_outputs' folder for charts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
