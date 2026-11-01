"""
DMVP 5 - Retail Sales Data Analysis
Analyses sales by region and product category.
Dataset required: Retail_Sales_Data.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("Retail_Sales_Data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Place the CSV file beside this script."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    required = {"Region", "Category", "Quantity", "Sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "The dataset does not contain the expected retail-sales columns: "
            + ", ".join(sorted(missing))
        )

    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Convert analysis fields to numeric values.
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Region", "Category", "Sales", "Quantity"]).copy()

    # Remove blank category/region labels and standardize their spacing.
    for column in ["Region", "Category"]:
        df[column] = df[column].astype(str).str.strip()
    df = df[(df["Region"] != "") & (df["Category"] != "")]

    if df.empty:
        raise ValueError("No usable rows remain after cleaning the dataset.")

    # Total sales for each region.
    sales_by_region = (
        df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    )
    print("\nTotal Sales by Region:")
    print(sales_by_region)

    top_region = sales_by_region.idxmax()
    print(f"\nTop-performing region: {top_region}")
    print(f"Total sales: {sales_by_region.max():,.2f}")

    plt.figure(figsize=(10, 5))
    sales_by_region.plot(kind="bar")
    plt.title("Total Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_region.png", dpi=150)
    plt.show()
    plt.close()

    # Sales by region and product category.
    sales_by_region_category = (
        df.pivot_table(
            index="Region",
            columns="Category",
            values="Sales",
            aggfunc="sum",
            fill_value=0,
        )
    )
    print("\nSales by Region and Product Category:")
    print(sales_by_region_category)

    ax = sales_by_region_category.plot(
        kind="bar", figsize=(11, 6), width=0.8
    )
    ax.set_title("Sales by Region and Product Category")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_region_category.png", dpi=150)
    plt.show()
    plt.close()

    sales_by_region.to_csv(OUTPUT_DIR / "sales_by_region.csv")
    sales_by_region_category.to_csv(
        OUTPUT_DIR / "sales_by_region_category.csv"
    )
    df.to_csv(OUTPUT_DIR / "retail_sales_cleaned.csv", index=False)
    print("\nAnalysis completed. Results saved in the outputs folder.")


if __name__ == "__main__":
    main()
