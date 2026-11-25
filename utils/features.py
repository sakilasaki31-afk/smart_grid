
import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "household_power_clean.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "household_power_features.csv"
)


# ==========================================
# STEP 2: LOAD DATA
# ==========================================

def load_data():

    print("Loading cleaned electricity dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)

    print("Dataset loaded:", df.shape)

    return df


# ==========================================
# STEP 3: TEMPORAL FEATURES
# ==========================================

def create_temporal_features(df):

    df["hour"] = df["datetime"].dt.hour

    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["month"] = df["datetime"].dt.month

    df["day"] = df["datetime"].dt.day

    df["year"] = df["datetime"].dt.year

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] < 6)
    ).astype(int)

    df["is_morning_peak"] = (
        (df["hour"] >= 6) &
        (df["hour"] <= 10)
    ).astype(int)

    df["is_evening_peak"] = (
        (df["hour"] >= 17) &
        (df["hour"] <= 22)
    ).astype(int)

    print("Temporal features created!")

    return df


# ==========================================
# STEP 4: CYCLICAL ENCODING
# ==========================================

def create_cyclical_features(df):

    # Hour encoding
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Day of week encoding
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Month encoding
    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    print("Cyclical features created!")

    return df


# ==========================================
# STEP 5: LAG FEATURES
# ==========================================

def create_lag_features(df):

    lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]

    columns = [
        "Global_active_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for col in columns:

        for lag in lag_periods:

            df[f"{col}_lag_{lag}"] = (
                df[col].shift(lag)
            )

    print("Lag features created!")

    return df


# ==========================================
# STEP 6: ROLLING STATISTICS
# ==========================================

def create_rolling_features(df):

    windows = [3, 6, 12, 24, 48]

    for window in windows:

        shifted_power = (
            df["Global_active_power"].shift(1)
        )

        rolling = shifted_power.rolling(
            window=window,
            min_periods=window
        )

        df[f"power_rolling_mean_{window}"] = (
            rolling.mean()
        )

        df[f"power_rolling_std_{window}"] = (
            rolling.std()
        )

        df[f"power_rolling_max_{window}"] = (
            rolling.max()
        )

        df[f"power_rolling_min_{window}"] = (
            rolling.min()
        )

    print("Rolling statistics created!")

    return df


# ==========================================
# STEP 7: INTERACTION FEATURES
# ==========================================

def create_interaction_features(df):

    # Reactive power / active power ratio
    df["reactive_active_ratio"] = (
        df["Global_reactive_power"] /
        df["Global_active_power"].replace(0, np.nan)
    )

    # Sub-metering ratio
    df["submeter_ratio"] = (
        df["Sub_metering_total"] /
        (df["Global_active_power"] * 1000).replace(0, np.nan)
    )

    # Voltage and power interaction
    df["voltage_power_interaction"] = (
        df["Voltage"] * df["Global_active_power"]
    )

    print("Interaction features created!")

    return df


# ==========================================
# STEP 8: ANOMALY DETECTION LABELS
# ==========================================

def create_anomaly_labels(df):

    # Calculate rolling mean and standard deviation
    # using only previous observations

    previous_power = df["Global_active_power"].shift(1)

    rolling_mean = previous_power.rolling(
        window=24,
        min_periods=6
    ).mean()

    rolling_std = previous_power.rolling(
        window=24,
        min_periods=6
    ).std()

    # Z-score
    df["power_zscore"] = (
        (df["Global_active_power"] - rolling_mean) /
        rolling_std.replace(0, np.nan)
    )

    # Zero reading
    df["zero_reading"] = (
        df["Global_active_power"] == 0
    ).astype(int)

    # Statistical outlier
    df["statistical_outlier"] = (
        df["power_zscore"].abs() > 3
    ).fillna(False).astype(int)

    # Extreme peak compared with previous 168 hours
    historical_max = (
        df["Global_active_power"]
        .shift(1)
        .rolling(window=168, min_periods=24)
        .max()
    )

    df["extreme_peak"] = (
        df["Global_active_power"] >
        historical_max * 1.5
    ).fillna(False).astype(int)

    # Final anomaly label
    df["is_anomaly"] = (
        (df["zero_reading"] == 1) |
        (df["statistical_outlier"] == 1) |
        (df["extreme_peak"] == 1)
    ).astype(int)

    # Anomaly category
    conditions = [
        df["zero_reading"] == 1,
        df["statistical_outlier"] == 1,
        df["extreme_peak"] == 1
    ]

    choices = [
        "zero_reading",
        "statistical_outlier",
        "extreme_peak"
    ]

    df["anomaly_type"] = np.select(
        conditions,
        choices,
        default="normal"
    )

    print("Anomaly labels created!")

    return df


# ==========================================
# STEP 9: FORECAST TARGET
# ==========================================

def create_forecast_target(df):

    # Predict next hour electricity consumption
    df["target_next_hour"] = (
        df["Global_active_power"].shift(-1)
    )

    print("Next-hour forecast target created!")

    return df


# ==========================================
# STEP 10: COMPLETE FEATURE PIPELINE
# ==========================================

def build_features():

    df = load_data()

    df = create_temporal_features(df)

    df = create_cyclical_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_interaction_features(df)

    df = create_anomaly_labels(df)

    df = create_forecast_target(df)

    # Remove rows with missing values created
    # by lag, rolling and target calculations
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    # Save feature-engineered dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n================================")
    print("FEATURE ENGINEERING COMPLETED!")
    print("================================")

    print("Final dataset shape:", df.shape)

    print("Total features:", len(df.columns))

    print("\nAnomaly distribution:")

    print(df["is_anomaly"].value_counts())

    print("\nAnomaly types:")

    print(df["anomaly_type"].value_counts())

    print("\nSaved file:", OUTPUT_FILE)

    print("\nFirst five records:")

    print(df.head())


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    build_features()
