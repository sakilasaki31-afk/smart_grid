
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "household_power_features.csv"
)

ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_next_hour"

# Files required for dashboard predictions
FEATURE_COLUMNS_FILE = ARTIFACT_DIR / "forecast_feature_columns.pkl"
FEATURE_MEDIANS_FILE = ARTIFACT_DIR / "forecast_feature_medians.pkl"


# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

def load_dataset():

    print("Loading electricity feature dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Feature dataset not found. Run utils/features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)

    print("Dataset shape:", df.shape)

    return df


# ==========================================
# STEP 3: PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(df):

    if TARGET not in df.columns:
        raise ValueError("Forecast target not found!")

    excluded_columns = [
        "datetime",
        "target_next_hour",
        "is_anomaly",
        "anomaly_type",
        "power_zscore",
        "zero_reading",
        "statistical_outlier",
        "extreme_peak"
    ]

    X = df.select_dtypes(
        include=[np.number]
    ).drop(
        columns=excluded_columns,
        errors="ignore"
    )

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    valid_rows = y.notna() & np.isfinite(y)

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("Number of input features:", X.shape[1])

    return X, y


# ==========================================
# STEP 4: CHRONOLOGICAL TRAIN TEST SPLIT
# ==========================================

def split_data(X, y):

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # Calculate medians using training data only
    medians = X_train.median()

    # Handle columns that contain only missing values
    medians = medians.fillna(0)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Save the exact training feature schema
    feature_columns = X_train.columns.tolist()

    joblib.dump(
        feature_columns,
        FEATURE_COLUMNS_FILE
    )

    joblib.dump(
        medians,
        FEATURE_MEDIANS_FILE
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Feature columns saved:", len(feature_columns))
    print("Feature medians saved.")

    return X_train, X_test, y_train, y_test


# ==========================================
# STEP 5: EVALUATION FUNCTION
# ==========================================

results = []

def evaluate_model(name, y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    non_zero = np.abs(y_true) > 1e-8

    mape = np.mean(
        np.abs(
            (y_true[non_zero] - predictions[non_zero])
            / y_true[non_zero]
        )
    ) * 100 if non_zero.any() else np.nan

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2,
        "MAPE_Percent": mape
    })

    print("\n================================")
    print("MODEL:", name)
    print("================================")

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    print("MAPE:", round(mape, 2), "%")

    return mae, rmse, r2, mape


# ==========================================
# STEP 6: BASELINE MODEL
# ==========================================

def evaluate_baseline(y_train, y_test):

    print("\nTraining baseline forecast...")

    predictions = X_test_global_power.copy()

    evaluate_model(
        "Naive Baseline",
        y_test,
        predictions
    )


# ==========================================
# STEP 7: TRAIN REGRESSION MODELS
# ==========================================

def train_regression_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    models = {

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=20,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    }

    fitted_models = {}

    for name, model in models.items():

        print("\nTraining:", name)

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        evaluate_model(
            name,
            y_test,
            predictions
        )

        fitted_models[name] = model

        model_filename = (
            name.lower().replace(" ", "_") + "_forecast.pkl"
        )

        joblib.dump(
            model,
            ARTIFACT_DIR / model_filename
        )

        print("Model saved:", model_filename)

    return fitted_models


# ==========================================
# STEP 8: SAVE COMPARISON RESULTS
# ==========================================

def save_results():

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAE",
        ascending=True
    )

    output_file = (
        ARTIFACT_DIR / "forecast_model_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n================================")
    print("FORECAST MODEL COMPARISON")
    print("================================")

    print(results_df.to_string(index=False))

    print("\nResults saved:", output_file)

    return results_df


# ==========================================
# STEP 9: MAIN EXECUTION
# ==========================================

def main():

    global X_test_global_power

    df = load_dataset()

    X, y = prepare_data(df)

    current_power = df.loc[
        y.index, "Global_active_power"
    ]

    split_index = int(len(X) * 0.80)

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    X_test_global_power = current_power.iloc[
        split_index:
    ].to_numpy()

    print("\nStarting baseline evaluation...")

    evaluate_baseline(
        y_train,
        y_test
    )

    print("\nStarting regression model training...")

    train_regression_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_results()

    print("\n================================")
    print("DEMAND FORECASTING COMPLETED!")
    print("================================")

    print("\nDashboard preprocessing files saved:")
    print(FEATURE_COLUMNS_FILE)
    print(FEATURE_MEDIANS_FILE)


if __name__ == "__main__":
    main()
