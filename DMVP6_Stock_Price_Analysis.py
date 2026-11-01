"""
DMVP 6 - Stock Price Time-Series Analysis
Analyses historical ETF prices and creates a basic ARIMA forecast for SPY.
Dataset required: Stock_Prices.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("Stock_Prices.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Place the CSV file beside this script."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    required = {"Date", "SPY", "DIA", "EEM", "EFA", "XLE"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing expected columns: " + ", ".join(sorted(missing))
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in df.columns.difference(["Date"]):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["Date", "SPY"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .set_index("Date")
    )

    if len(df) < 40:
        raise ValueError("At least 40 valid SPY observations are recommended.")

    print("Dataset shape after cleaning:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nSPY price statistics:")
    print(df["SPY"].describe())

    # Historical SPY price.
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["SPY"], label="SPY Price")
    plt.title("Historical SPY Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spy_historical_price.png", dpi=150)
    plt.show()
    plt.close()

    # 20-observation moving average (approximately 20 trading days).
    df["SPY_MA_20"] = df["SPY"].rolling(window=20, min_periods=20).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["SPY"], label="SPY Price")
    plt.plot(df.index, df["SPY_MA_20"], label="20-Day Moving Average")
    plt.title("SPY Price with 20-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spy_moving_average.png", dpi=150)
    plt.show()
    plt.close()

    # Average price by calendar month across all years in the dataset.
    monthly_spy = df.groupby(df.index.month)["SPY"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_spy.index, monthly_spy.values, marker="o")
    plt.title("Average SPY Price by Calendar Month (All Years)")
    plt.xlabel("Calendar Month")
    plt.ylabel("Average SPY Price")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spy_monthly_average.png", dpi=150)
    plt.show()
    plt.close()

    # Price-level correlation, matching the original practical.
    selected_prices = df[["SPY", "DIA", "EEM", "EFA", "XLE"]]
    price_corr = selected_prices.corr()
    print("\nCorrelation Matrix (price levels):")
    print(price_corr)

    # Daily-return correlation is often more informative for co-movement.
    daily_returns = selected_prices.pct_change().dropna()
    return_corr = daily_returns.corr()
    print("\nCorrelation Matrix (daily percentage returns):")
    print(return_corr)

    plt.figure(figsize=(8, 6))
    plt.imshow(return_corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(return_corr.columns)), return_corr.columns)
    plt.yticks(range(len(return_corr.columns)), return_corr.columns)
    plt.title("Correlation of Daily ETF Returns")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_return_correlation.png", dpi=150)
    plt.show()
    plt.close()

    # ARIMA(1, 1, 1) forecast: next 30 observations, not guaranteed returns.
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "Install statsmodels with: pip install statsmodels"
        ) from exc

    model = ARIMA(df["SPY"], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    print("\nARIMA(1,1,1) forecast for the next 30 trading observations:")
    print(forecast)

    plt.figure(figsize=(12, 5))
    history_tail = df["SPY"].tail(120)
    plt.plot(
        range(len(history_tail)),
        history_tail.values,
        label="Recent Historical SPY Price",
    )
    forecast_x = range(len(history_tail), len(history_tail) + len(forecast))
    plt.plot(forecast_x, forecast.values, label="30-Step Forecast")
    plt.title("SPY Price Forecast Using ARIMA(1,1,1)")
    plt.xlabel("Observation (historical followed by forecast)")
    plt.ylabel("SPY Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spy_arima_forecast.png", dpi=150)
    plt.show()
    plt.close()

    df.to_csv(OUTPUT_DIR / "stock_prices_analysis.csv")
    forecast.rename("Forecast_SPY").to_csv(
        OUTPUT_DIR / "spy_30_step_forecast.csv"
    )
    print("\nAnalysis completed. Results saved in the outputs folder.")
    print("Note: This educational forecast is not financial advice.")


if __name__ == "__main__":
    main()
