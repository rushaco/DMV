"""
DMVP 3 - Telecom Customer Churn Data Analysis
Performs dataset inspection, cleaning, categorical summaries, outlier
identification, and churn-label preparation.

This script prepares data for later analysis/modeling; it does not train
a machine-learning model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("Telecom_Customer_Churn.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"{DATA_FILE} was not found. Place it in the project folder."
    )

df = pd.read_csv(DATA_FILE)
print("First five records:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# 2. Standardize column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)

# 3. Identify and remove empty/unnamed export columns.
unnamed_columns = [
    column for column in df.columns
    if column.lower().startswith("unnamed:")
]
if unnamed_columns:
    print("\nRemoving export-only columns:", unnamed_columns)
    df = df.drop(columns=unnamed_columns)

# 4. Inspect missing values and duplicate records
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate records before:", df.duplicated().sum())
df = df.drop_duplicates().copy()
print("Duplicate records after:", df.duplicated().sum())

# 5. Inspect categorical distributions
categorical_columns = df.select_dtypes(include=["object", "string"]).columns
for column in categorical_columns:
    print(f"\nTop values in {column}:")
    print(df[column].value_counts(dropna=False).head(10))

# 6. Convert numeric fields safely
numeric_candidates = [
    "Age", "Number_of_Dependents", "Zip_Code", "Number_of_Referrals",
    "Monthly_Charge", "Total_Charges", "Total_Refunds",
    "Total_Extra_Data_Charges", "Total_Revenue"
]
for column in numeric_candidates:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# 7. Handle missing values without inventing customer service usage.
# Missing categories are retained as "Unknown"; this is safer than assuming
# that a missing value means "No".
categorical_columns = df.select_dtypes(include=["object", "string"]).columns
for column in categorical_columns:
    df[column] = df[column].fillna("Unknown").astype("string").str.strip()

# Numeric missing values are imputed with the median and reported.
numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    if df[column].isna().any():
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

print("\nTotal missing values after cleaning:", df.isnull().sum().sum())

# 8. Check suspicious values rather than blindly capping all numeric columns.
if "Monthly_Charge" in df.columns:
    invalid_monthly = df["Monthly_Charge"] < 0
    print("Negative Monthly_Charge records:", int(invalid_monthly.sum()))

# 9. Identify IQR outliers in suitable continuous financial/age variables.
# ZIP codes, IDs, and count variables are deliberately excluded.
outlier_candidates = [
    column for column in
    ["Age", "Monthly_Charge", "Total_Charges", "Total_Refunds",
     "Total_Extra_Data_Charges", "Total_Revenue"]
    if column in df.columns
]

outlier_summary = []
for column in outlier_candidates:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[column] < lower) | (df[column] > upper)).sum())
    outlier_summary.append({
        "Column": column,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": count,
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\nIQR outlier summary (records are not automatically removed):")
print(outlier_df)
outlier_df.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

# 10. Prepare a churn target carefully.
# "Joined" is a separate customer status, not the same as "Stayed".
if "Customer_Status" in df.columns:
    status = df["Customer_Status"].astype("string").str.strip().str.lower()
    df["Is_Churned"] = status.map({"churned": 1, "stayed": 0})
    print("\nCustomer status distribution:")
    print(df["Customer_Status"].value_counts(dropna=False))
    print("\nRows eligible for a binary Churned/Stayed target:",
          int(df["Is_Churned"].notna().sum()))
    print("Rows with other/unknown status:",
          int(df["Is_Churned"].isna().sum()))

# 11. Visualize customer status distribution
if "Customer_Status" in df.columns:
    status_counts = df["Customer_Status"].value_counts()
    plt.figure(figsize=(7, 5))
    status_counts.plot(kind="bar")
    plt.title("Customer Status Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_status.png", dpi=150)
    plt.show()

# 12. Visualize churn reasons where available
if "Churn_Reason" in df.columns:
    reasons = df.loc[
        ~df["Churn_Reason"].isin(["Unknown", ""]),
        "Churn_Reason"
    ].value_counts().head(10)

    if not reasons.empty:
        plt.figure(figsize=(10, 6))
        reasons.sort_values().plot(kind="barh")
        plt.title("Top Customer Churn Reasons")
        plt.xlabel("Number of Customers")
        plt.ylabel("Churn Reason")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_churn_reasons.png", dpi=150)
        plt.show()

# 13. Save cleaned data
df.to_csv(OUTPUT_DIR / "Cleaned_Telecom_Customer_Churn.csv", index=False)
print("\nFinal dataset shape:", df.shape)
print("Final missing values:", int(df.isnull().sum().sum()))
print("Final duplicate records:", int(df.duplicated().sum()))
print("\nCleaned telecom dataset saved successfully.")
