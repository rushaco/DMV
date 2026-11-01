"""
DMVP 7 - Air Quality Analysis
Studies AQI and pollutant trends for Ahmedabad.
Dataset required: city_day.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("city_day.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()
    plt.close()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Place the CSV file beside this script."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    required = {"City", "Date", "AQI", "PM2.5", "PM10", "CO"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing expected columns: " + ", ".join(sorted(missing))
        )

    print("Original dataset shape:", df.shape)
    print("\nMissing values in original dataset:")
    print(df.isnull().sum())

    # Select Ahmedabad and parse its date values.
    df = df[df["City"].astype(str).str.strip().str.casefold() == "ahmedabad"].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    pollutant_columns = [
        column for column in df.columns
        if column not in {"City", "Date", "AQI_Bucket"}
    ]
    for column in pollutant_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep records with the key fields used in the practical.
    df = df.dropna(subset=["Date", "AQI", "PM2.5", "PM10", "CO"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])

    if df.empty:
        raise ValueError(
            "No Ahmedabad rows remain with valid Date, AQI, PM2.5, PM10 and CO."
        )

    print("\nAhmedabad data after cleaning:")
    print(df.head())
    print("Cleaned shape:", df.shape)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # AQI trend.
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["AQI"])
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("AQI Trend Over Time - Ahmedabad")
    plt.xticks(rotation=45)
    save_and_show("ahmedabad_aqi_trend.png")

    # Individual pollutant trends.
    for pollutant in ["PM2.5", "PM10", "CO"]:
        plt.figure(figsize=(10, 4))
        plt.plot(df["Date"], df[pollutant])
        plt.xlabel("Date")
        plt.ylabel(pollutant)
        plt.title(f"{pollutant} Trend Over Time - Ahmedabad")
        plt.xticks(rotation=45)
        save_and_show(f"ahmedabad_{pollutant.lower().replace('.', '')}_trend.png")

    # Compare AQI for the first ten available cleaned dates.
    sample = df.head(10)
    plt.figure(figsize=(9, 4))
    plt.bar(sample["Date"].dt.strftime("%d-%m-%Y"), sample["AQI"])
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("AQI Comparison - First 10 Available Ahmedabad Records")
    plt.xticks(rotation=45, ha="right")
    save_and_show("ahmedabad_aqi_first_10.png")

    # AQI distribution.
    plt.figure(figsize=(7, 5))
    plt.boxplot(df["AQI"].dropna())
    plt.ylabel("AQI")
    plt.title("AQI Distribution - Ahmedabad")
    plt.grid(True)
    save_and_show("ahmedabad_aqi_boxplot.png")

    # Relationship between PM2.5 and AQI.
    plt.figure(figsize=(7, 5))
    plt.scatter(df["PM2.5"], df["AQI"], alpha=0.6)
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    plt.title("PM2.5 vs AQI - Ahmedabad")
    save_and_show("ahmedabad_pm25_vs_aqi.png")

    print("\nAQI summary statistics:")
    print(df["AQI"].describe())

    df.to_csv(OUTPUT_DIR / "ahmedabad_air_quality_cleaned.csv", index=False)
    print("\nAnalysis completed. Charts and cleaned data saved in outputs.")


if __name__ == "__main__":
    main()
