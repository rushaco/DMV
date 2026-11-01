"""
DMVP 4 - Real Estate Price Data Analysis
Performs data inspection, cleaning, location filtering, price summaries,
IQR outlier identification, and export of the cleaned dataset.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("RealEstate_Prices.csv")
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

# 2. Clean column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "_")
    .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
)

# 3. Convert date column into a datetime value
if "date" in df.columns:
    df["date"] = pd.to_datetime(
        df["date"].astype("string"),
        format="%Y%m%dT%H%M%S",
        errors="coerce"
    )

# 4. Inspect and handle missing/duplicate records
print("\nMissing values before cleaning:")
print(df.isnull().sum())
print("\nDuplicate records before:", df.duplicated().sum())

df = df.drop_duplicates().copy()

# Do not blindly fill missing values in every column. For this dataset,
# remove rows missing the target price and report remaining missing values.
if "price" not in df.columns:
    raise KeyError("The dataset must contain a 'price' column.")

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"]).copy()

print("\nMissing values after basic cleaning:")
print(df.isnull().sum())

# 5. Inspect numeric and categorical variables
print("\nDescriptive statistics:")
print(df.describe(include="all"))

categorical_columns = df.select_dtypes(include=["object", "string"]).columns
print("\nCategorical columns:", categorical_columns.tolist())

# 6. Filter properties by a selected ZIP code
if "zipcode" in df.columns:
    selected_zipcode = df["zipcode"].iloc[0]
    filtered_df = df[df["zipcode"] == selected_zipcode].copy()
    print("\nSelected ZIP code:", selected_zipcode)
    print("Number of properties:", len(filtered_df))
    print(filtered_df.head())
    filtered_df.to_csv(OUTPUT_DIR / "Properties_By_Zipcode.csv", index=False)

# 7. Calculate average sale price by ZIP code
if "zipcode" in df.columns:
    average_price_by_zipcode = (
        df.groupby("zipcode")["price"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nTop 10 ZIP codes by average sale price:")
    print(average_price_by_zipcode.head(10))
    average_price_by_zipcode.to_csv(
        OUTPUT_DIR / "Average_Price_By_Zipcode.csv"
    )

# 8. Identify price outliers using the IQR method.
# Outliers are flagged and reported, not automatically deleted.
q1 = df["price"].quantile(0.25)
q3 = df["price"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (df["price"] < lower_bound) | (df["price"] > upper_bound)
print("\nPrice outlier analysis:")
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of potential outliers:", int(outlier_mask.sum()))

df["Price_Outlier"] = outlier_mask
df.loc[outlier_mask].to_csv(OUTPUT_DIR / "Potential_Price_Outliers.csv",
                            index=False)

# 9. Visualize the distribution of property prices
plt.figure(figsize=(9, 5))
plt.hist(df["price"], bins=40, edgecolor="black")
plt.title("Distribution of Real Estate Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 4))
plt.boxplot(df["price"], vert=True)
plt.title("Real Estate Price Outlier Analysis")
plt.ylabel("Sale Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_boxplot.png", dpi=150)
plt.show()

# 10. Save the cleaned dataset
df.to_csv(OUTPUT_DIR / "Cleaned_RealEstate_Prices.csv", index=False)
print("\nFinal dataset shape:", df.shape)
print("Final missing values:", int(df.isnull().sum().sum()))
print("Final duplicate records:", int(df.duplicated().sum()))
print("\nReal estate analysis completed.")
