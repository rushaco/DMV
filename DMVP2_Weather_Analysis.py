"""
DMVP 2 - Weather Data Analysis using OpenWeatherMap API
Fetches forecast data, performs basic preprocessing and statistical analysis,
and creates visualizations.

Before running:
1. Create an API key at https://openweathermap.org/api
2. Set it as an environment variable named OPENWEATHER_API_KEY.
3. Never commit your API key to GitHub.
"""

import os
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt

CITY = "Pune"
COUNTRY = "IN"
API_URL = "https://api.openweathermap.org/data/2.5/forecast"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_weather_forecast(city: str, country: str) -> tuple[dict, pd.DataFrame]:
    """Fetch forecast data and return the API response and a DataFrame."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENWEATHER_API_KEY is not set. Add your API key as an environment variable."
        )

    params = {
        "q": f"{city},{country}",
        "appid": api_key,
        "units": "metric",
    }

    response = requests.get(API_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if "list" not in data or not data["list"]:
        raise ValueError("The API response did not contain forecast records.")

    records = []
    for item in data["list"]:
        records.append({
            "Date": item.get("dt_txt"),
            "Temperature": item.get("main", {}).get("temp"),
            "Humidity": item.get("main", {}).get("humidity"),
            "Wind_Speed": item.get("wind", {}).get("speed"),
            # OpenWeatherMap 'pop' is a probability from 0 to 1.
            "Rain_Probability": item.get("pop", 0),
        })

    weather_df = pd.DataFrame(records)
    return data, weather_df


# 1. Fetch forecast data
try:
    data, df = fetch_weather_forecast(CITY, COUNTRY)
except requests.exceptions.RequestException as error:
    raise SystemExit(f"Weather API request failed: {error}")
except (EnvironmentError, ValueError) as error:
    raise SystemExit(str(error))

print("City information:")
print(data.get("city", {}))
print("\nFirst five forecast records:")
print(df.head())

# 2. Data preprocessing
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "Temperature", "Humidity", "Wind_Speed", "Rain_Probability"
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["Date"]).drop_duplicates().sort_values("Date")
df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

# Keep rain probability within its documented range.
df["Rain_Probability"] = df["Rain_Probability"].clip(0, 1)

print("\nDataset information:")
print(df.info())
print("\nDescriptive statistics:")
print(df[numeric_columns].describe())
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# 3. Basic temperature analysis
print("\nAverage temperature:", round(df["Temperature"].mean(), 2), "°C")
print("Minimum temperature:", round(df["Temperature"].min(), 2), "°C")
print("Maximum temperature:", round(df["Temperature"].max(), 2), "°C")

# 4. Temperature trend
plt.figure(figsize=(11, 5))
plt.plot(df["Date"], df["Temperature"], marker="o", linewidth=1.5)
plt.title(f"Temperature Forecast - {CITY}")
plt.xlabel("Date and Time")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=35, ha="right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temperature_trend.png", dpi=150)
plt.show()

# 5. Humidity and wind speed trends (separate axes because units differ)
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(df["Date"], df["Humidity"], marker="o", label="Humidity")
ax1.set_xlabel("Date and Time")
ax1.set_ylabel("Humidity (%)")
ax1.tick_params(axis="x", rotation=35)
ax2 = ax1.twinx()
ax2.plot(df["Date"], df["Wind_Speed"], marker="s", label="Wind speed")
ax2.set_ylabel("Wind speed (m/s)")
plt.title("Humidity and Wind Speed Forecast")
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "humidity_wind_trend.png", dpi=150)
plt.show()

# 6. Rain probability
plt.figure(figsize=(11, 5))
plt.bar(df["Date"], df["Rain_Probability"] * 100, width=0.08)
plt.title("Forecast Probability of Precipitation")
plt.xlabel("Date and Time")
plt.ylabel("Rain probability (%)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rain_probability.png", dpi=150)
plt.show()

# 7. Daily average weather conditions
daily = (
    df.set_index("Date")[["Temperature", "Humidity", "Wind_Speed"]]
    .resample("D")
    .mean()
)
print("\nDaily average weather conditions:")
print(daily)

# 8. Correlation analysis
correlation = df[numeric_columns].corr()
print("\nWeather attribute correlation:")
print(correlation)

plt.figure(figsize=(7, 5))
image = plt.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(image, label="Correlation")
plt.xticks(range(len(correlation.columns)), correlation.columns,
           rotation=35, ha="right")
plt.yticks(range(len(correlation.columns)), correlation.columns)
for row in range(len(correlation.columns)):
    for column in range(len(correlation.columns)):
        plt.text(column, row, f"{correlation.iloc[row, column]:.2f}",
                 ha="center", va="center")
plt.title("Weather Attribute Correlation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weather_correlation.png", dpi=150)
plt.show()

# 9. Plot the geographical coordinates returned by the API
longitude = data["city"]["coord"]["lon"]
latitude = data["city"]["coord"]["lat"]

plt.figure(figsize=(7, 5))
plt.scatter(longitude, latitude, s=100)
plt.annotate(data["city"]["name"], (longitude, latitude),
             xytext=(6, 6), textcoords="offset points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Location of the Selected City")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "city_location.png", dpi=150)
plt.show()

# 10. Save processed forecast data
df.to_csv(OUTPUT_DIR / "Processed_Weather_Data.csv", index=False)
daily.to_csv(OUTPUT_DIR / "Daily_Weather_Summary.csv")
print("\nWeather analysis completed. Results are saved in the outputs folder.")
