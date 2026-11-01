"""
DMVP 1 - Sales Data Analysis
Demonstrates data creation, CSV/Excel/JSON handling, cleaning,
summary statistics, aggregation, and visualization.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Create a sample sales dataset
data = {
    "Order_ID": list(range(101, 111)),
    "Product_Category": [
        "Electronics", "Clothing", "Grocery", "Furniture", "Books",
        "Sports", "Electronics", "Clothing", "Grocery", "Furniture"
    ],
    "Quantity": [2, 3, 5, 1, 4, 2, 1, 5, 3, 2],
    "Price": [50000, 1500, 200, 12000, 500, 3000, 25000, 2000, 300, 15000],
}

df = pd.DataFrame(data)
print("Original dataset:")
print(df)

# 2. Export the dataset into different file formats
df.to_csv(OUTPUT_DIR / "Sales_Data.csv", index=False)
df.to_excel(OUTPUT_DIR / "Sales_Data.xlsx", index=False)
df.to_json(OUTPUT_DIR / "Sales_Data.json", orient="records", indent=4)
print("\nCSV, Excel and JSON files created successfully.")

# 3. Read the exported files and verify their contents
csv_data = pd.read_csv(OUTPUT_DIR / "Sales_Data.csv")
excel_data = pd.read_excel(OUTPUT_DIR / "Sales_Data.xlsx")
json_data = pd.read_json(OUTPUT_DIR / "Sales_Data.json")

print("\nCSV preview:")
print(csv_data.head())
print("\nExcel preview:")
print(excel_data.head())
print("\nJSON preview:")
print(json_data.head())

# 4. Inspect and clean the dataset
print("\nDataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()
df["Product_Category"] = (
    df["Product_Category"].astype("string").str.strip().str.title()
)

# Validate values before calculating sales
if (df["Quantity"] < 0).any() or (df["Price"] < 0).any():
    raise ValueError("Quantity and Price must not contain negative values.")

# 5. Calculate total sales for each order
df["Total_Sales"] = df["Quantity"] * df["Price"]
print("\nSales dataset after cleaning and calculation:")
print(df)

# 6. Generate descriptive statistics and business summaries
print("\nDescriptive statistics:")
print(df[["Quantity", "Price", "Total_Sales"]].describe())

print("\nTotal sales:", df["Total_Sales"].sum())
print("Average order value:", df["Total_Sales"].mean())

category_sales = (
    df.groupby("Product_Category")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)
print("\nSales by product category:")
print(category_sales)

category_count = df["Product_Category"].value_counts()

# 7. Visualize sales by category
plt.figure(figsize=(9, 5))
category_sales.plot(kind="bar")
plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_by_category.png", dpi=150)
plt.show()

# 8. Visualize order distribution by category
plt.figure(figsize=(7, 7))
category_count.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    ylabel=""
)
plt.title("Order Distribution by Product Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_distribution.png", dpi=150)
plt.show()

# 9. Identify the spread of order sales
plt.figure(figsize=(7, 4))
plt.boxplot(df["Total_Sales"], vert=True)
plt.title("Distribution of Total Sales")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_boxplot.png", dpi=150)
plt.show()

# 10. Save the final processed dataset
df.to_csv(OUTPUT_DIR / "Processed_Sales_Data.csv", index=False)
print("\nAnalysis completed. Processed data and charts are in the outputs folder.")
