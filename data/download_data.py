
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "raw" / "household_power_consumption.txt"
OUTPUT_FILE = BASE_DIR / "processed" / "household_power_clean.csv"


def load_and_clean_data():

    print("Loading electricity dataset...")

    # Check whether dataset exists
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    # Read original dataset
    df = pd.read_csv(
        RAW_FILE,
        sep=";",
        na_values=["?"],
        low_memory=False,
        encoding="latin-1"
    )

    print("Original dataset shape:", df.shape)

    # Combine Date and Time
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["datetime"])

    # Convert measurement columns into numeric values
    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Set datetime as index
    df = df.set_index("datetime")

    # Remove original date and time columns
    df = df.drop(columns=["Date", "Time"])

    # Sort records chronologically
    df = df.sort_index()

    print("Converting minute-level data into hourly data...")

    # Hourly aggregation
    aggregation = {
        "Global_active_power": "mean",
        "Global_reactive_power": "mean",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum"
    }

    hourly_df = df.resample("h").agg(aggregation)

    # Handle missing values using time interpolation
    hourly_df = hourly_df.interpolate(
        method="time",
        limit=3
    )

    # Remove rows with remaining missing values
    hourly_df = hourly_df.dropna()

    # Derived features
    hourly_df["Sub_metering_total"] = (
        hourly_df["Sub_metering_1"]
        + hourly_df["Sub_metering_2"]
        + hourly_df["Sub_metering_3"]
    )

    # Estimate unmetered consumption in Wh per hour
    hourly_df["Unmetered"] = (
        hourly_df["Global_active_power"] * 1000
        - hourly_df["Sub_metering_total"]
    )

    # Reset datetime index
    hourly_df = hourly_df.reset_index()

    # Save processed dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    hourly_df.to_csv(OUTPUT_FILE, index=False)

    print("\nData preprocessing completed!")
    print("Cleaned dataset shape:", hourly_df.shape)
    print("Saved file:", OUTPUT_FILE)
    print("\nFirst five records:")
    print(hourly_df.head())


if __name__ == "__main__":
    load_and_clean_data()
