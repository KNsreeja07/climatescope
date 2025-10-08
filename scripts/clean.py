import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("../data/GlobalWeatherRepository.csv")

# ---------- Handle Missing Values ----------
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after cleaning:\n", df.isnull().sum())

# ---------- Convert Units ----------
# Ensure temperature is consistent
if "temperature_fahrenheit" in df.columns and "temperature_celsius" in df.columns:
    df["temperature_celsius_calc"] = (df["temperature_fahrenheit"] - 32) * 5/9
    # Replace with calculated if mismatch
    df["temperature_celsius"] = np.where(
        np.isclose(df["temperature_celsius"], df["temperature_celsius_calc"], atol=0.5),
        df["temperature_celsius"],
        df["temperature_celsius_calc"]
    )
    df.drop(columns=["temperature_celsius_calc"], inplace=True)

# Pressure conversion check
if "pressure_in" in df.columns and "pressure_mb" in df.columns:
    df["pressure_mb_calc"] = df["pressure_in"] * 33.8639
    df["pressure_mb"] = np.where(
        np.isclose(df["pressure_mb"], df["pressure_mb_calc"], atol=1),
        df["pressure_mb"],
        df["pressure_mb_calc"]
    )
    df.drop(columns=["pressure_mb_calc"], inplace=True)

# ---------- Normalize numeric values ----------
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ---------- Date Parsing ----------
df["date"] = pd.to_datetime(df["last_updated"]).dt.date

# ---------- Aggregate Data (daily averages) ----------
daily_avg = df.groupby(["country", "location_name", "date"])[numeric_cols].mean().reset_index()

print("Dataset shape after cleaning:", df.shape)
print("Daily averages shape:", daily_avg.shape)

# Save cleaned data
df.to_csv("../data/GlobalWeatherRepository_cleaned.csv", index=False)
daily_avg.to_csv("../data/GlobalWeatherRepository_daily.csv", index=False)

print("✅ Cleaning completed. Cleaned datasets saved.")
