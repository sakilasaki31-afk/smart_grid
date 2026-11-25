
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from pathlib import Path


# =====================================
# STEP 1: PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "artifacts"

CLEAN_FILE = DATA_DIR / "household_power_clean.csv"
FEATURE_FILE = DATA_DIR / "household_power_features.csv"

ANOMALY_MODEL = MODEL_DIR / "xgboost.pkl"
FORECAST_MODEL = MODEL_DIR / "lightgbm_forecast.pkl"

ANOMALY_COLUMNS = MODEL_DIR / "anomaly_feature_columns.pkl"
ANOMALY_MEDIANS = MODEL_DIR / "anomaly_feature_medians.pkl"

FORECAST_COLUMNS = MODEL_DIR / "forecast_feature_columns.pkl"
FORECAST_MEDIANS = MODEL_DIR / "forecast_feature_medians.pkl"


# =====================================
# STEP 2: PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Grid Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Grid Intelligence")
st.caption("Electricity Consumption Analytics Platform")


# =====================================
# STEP 3: LOAD DATA
# =====================================

@st.cache_data
def load_data():

    df = pd.read_csv(CLEAN_FILE)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_features():

    df = pd.read_csv(FEATURE_FILE)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

    return df


# =====================================
# STEP 4: LOAD MODELS AND PREPROCESSING
# =====================================

@st.cache_resource
def load_models():

    required_files = [
        ANOMALY_MODEL,
        FORECAST_MODEL,
        ANOMALY_COLUMNS,
        ANOMALY_MEDIANS,
        FORECAST_COLUMNS,
        FORECAST_MEDIANS
    ]

    missing_files = [
        file.name for file in required_files
        if not file.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing_files)
        )

    return {
        "anomaly_model": joblib.load(ANOMALY_MODEL),
        "forecast_model": joblib.load(FORECAST_MODEL),

        "anomaly_columns": joblib.load(ANOMALY_COLUMNS),
        "anomaly_medians": joblib.load(ANOMALY_MEDIANS),

        "forecast_columns": joblib.load(FORECAST_COLUMNS),
        "forecast_medians": joblib.load(FORECAST_MEDIANS)
    }


# =====================================
# STEP 5: PREPARE MODEL INPUT
# =====================================

def prepare_model_input(
    data,
    feature_columns,
    feature_medians
):

    # Use exactly the columns and order used in training
    X = data.reindex(
        columns=feature_columns
    ).copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    medians = feature_medians.reindex(
        feature_columns
    ).fillna(0)

    X = X.fillna(medians)

    return X


# =====================================
# STEP 6: LOAD APPLICATION DATA
# =====================================

if not CLEAN_FILE.exists():
    st.error("Cleaned dataset not found!")
    st.stop()

if not FEATURE_FILE.exists():
    st.error("Feature dataset not found!")
    st.stop()

try:
    df = load_data()
    features = load_features()
    models = load_models()

except Exception as error:
    st.error(f"Unable to load application data or models: {error}")
    st.stop()


# =====================================
# STEP 7: SIDEBAR NAVIGATION
# =====================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Anomaly Detection",
        "Demand Forecasting",
        "Model Comparison",
        "Dataset Explorer"
    ]
)


# =====================================
# PAGE 1: DASHBOARD OVERVIEW
# =====================================

if page == "Dashboard Overview":

    st.header("Electricity Consumption Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Average Active Power",
        f"{df['Global_active_power'].mean():.3f} kW"
    )

    col3.metric(
        "Average Voltage",
        f"{df['Voltage'].mean():.2f} V"
    )

    st.subheader("Electricity Consumption Trend")

    chart_data = df.tail(2000)

    fig = px.line(
        chart_data,
        x="datetime",
        y="Global_active_power",
        title="Electricity Consumption Over Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Hourly Consumption")

    hourly = df.copy()
    hourly["hour"] = hourly["datetime"].dt.hour

    hourly_data = (
        hourly.groupby("hour")["Global_active_power"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        hourly_data,
        x="hour",
        y="Global_active_power",
        title="Average Electricity Consumption by Hour"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================
# PAGE 2: ANOMALY DETECTION
# =====================================

elif page == "Anomaly Detection":

    st.header("⚠️ Electricity Anomaly Detection")

    st.write(
        "XGBoost predicts whether each feature-engineered "
        "record is normal or anomalous."
    )

    with st.spinner("Running XGBoost anomaly predictions..."):

        X_anomaly = prepare_model_input(
            features,
            models["anomaly_columns"],
            models["anomaly_medians"]
        )

        anomaly_model = models["anomaly_model"]

        predictions = anomaly_model.predict(X_anomaly)

        probabilities = anomaly_model.predict_proba(
            X_anomaly
        )[:, 1]

    anomaly_results = features.copy()

    anomaly_results["Predicted_Anomaly"] = predictions
    anomaly_results["Anomaly_Probability"] = probabilities

    anomaly_count = int((predictions == 1).sum())
    normal_count = int((predictions == 0).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Normal Records", f"{normal_count:,}")
    col2.metric("Predicted Anomalies", f"{anomaly_count:,}")
    col3.metric(
        "Anomaly Percentage",
        f"{(anomaly_count / len(predictions)) * 100:.2f}%"
    )

    st.success("XGBoost anomaly predictions completed!")

    st.subheader("Predicted Anomaly Distribution")

    distribution = pd.DataFrame({
        "Status": ["Normal", "Anomaly"],
        "Count": [normal_count, anomaly_count]
    })

    fig = px.pie(
        distribution,
        names="Status",
        values="Count",
        title="XGBoost Prediction Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Predicted Anomaly Records")

    anomaly_records = anomaly_results[
        anomaly_results["Predicted_Anomaly"] == 1
    ].sort_values(
        by="Anomaly_Probability",
        ascending=False
    )

    display_columns = [
        col for col in [
            "datetime",
            "Global_active_power",
            "Voltage",
            "Predicted_Anomaly",
            "Anomaly_Probability",
            "is_anomaly",
            "anomaly_type"
        ]
        if col in anomaly_records.columns
    ]

    st.dataframe(
        anomaly_records[display_columns].head(100),
        use_container_width=True
    )

    st.caption(
        "Predictions identify patterns similar to the "
        "rule-generated training labels. They do not prove "
        "electricity theft or equipment faults."
    )


# =====================================
# PAGE 3: DEMAND FORECASTING
# =====================================

elif page == "Demand Forecasting":

    st.header("⚡ Electricity Demand Forecasting")

    st.write(
        "LightGBM estimates next-hour electricity consumption "
        "using the latest available feature-engineered records."
    )

    with st.spinner("Running LightGBM forecasting..."):

        X_forecast = prepare_model_input(
            features,
            models["forecast_columns"],
            models["forecast_medians"]
        )

        forecast_model = models["forecast_model"]

        forecast_predictions = forecast_model.predict(
            X_forecast
        )

    forecast_results = features.copy()

    forecast_results["Predicted_Next_Hour"] = (
        forecast_predictions
    )

    latest_prediction = float(forecast_predictions[-1])

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Next-Hour Active Power",
        f"{latest_prediction:.3f} kW"
    )

    col2.metric(
        "Records Forecasted",
        f"{len(forecast_predictions):,}"
    )

    st.success("LightGBM forecasting completed!")

    if "datetime" in forecast_results.columns:
        latest_datetime = forecast_results["datetime"].iloc[-1]

        st.write(
            "Latest available input record:",
            latest_datetime
        )

    st.subheader("Actual vs Predicted Consumption")

    if "target_next_hour" in forecast_results.columns:

        chart_data = forecast_results.tail(500)

        fig = px.line(
            chart_data,
            x="datetime" if "datetime" in chart_data.columns else None,
            y=[
                "target_next_hour",
                "Predicted_Next_Hour"
            ],
            title="Actual and Predicted Next-Hour Consumption",
            labels={
                "value": "Active Power (kW)",
                "variable": "Measurement"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Recent Forecast Results")

    display_columns = [
        col for col in [
            "datetime",
            "Global_active_power",
            "target_next_hour",
            "Predicted_Next_Hour"
        ]
        if col in forecast_results.columns
    ]

    st.dataframe(
        forecast_results[display_columns].tail(100),
        use_container_width=True
    )

    st.caption(
        "The latest prediction estimates the next-hour target "
        "for the latest available feature record. It is not "
        "a live sensor reading."
    )


# =====================================
# PAGE 4: MODEL COMPARISON
# =====================================

elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")

    anomaly_file = (
        MODEL_DIR / "anomaly_model_comparison.csv"
    )

    forecast_file = (
        MODEL_DIR / "forecast_model_comparison.csv"
    )

    if anomaly_file.exists():

        anomaly_results = pd.read_csv(anomaly_file)

        st.subheader("Anomaly Detection Models")

        st.dataframe(
            anomaly_results,
            use_container_width=True
        )

        fig = px.bar(
            anomaly_results,
            x="Model",
            y="F1_Score",
            title="Classification F1 Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if forecast_file.exists():

        forecast_results = pd.read_csv(forecast_file)

        st.subheader("Forecasting Models")

        st.dataframe(
            forecast_results,
            use_container_width=True
        )

        fig = px.bar(
            forecast_results,
            x="Model",
            y="MAE",
            title="Regression MAE Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =====================================
# PAGE 5: DATASET EXPLORER
# =====================================

elif page == "Dataset Explorer":

    st.header("Household Electricity Dataset")

    col1, col2 = st.columns(2)

    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", f"{len(df.columns):,}")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        file_name="household_power_clean.csv",
        mime="text/csv"
    )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Smart Grid Intelligence | "
    "Python | SQL | Machine Learning | Streamlit"
)
