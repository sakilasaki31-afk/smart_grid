# ⚡ Smart Grid Intelligence – Electricity Consumption Analytics Platform

## 📌 Project Overview

Smart Grid Intelligence is a Machine Learning-based electricity consumption analytics platform designed to analyze household electricity usage, detect abnormal consumption patterns, and forecast future electricity demand.

The project uses Python, SQL-related data processing concepts, Machine Learning, and Streamlit to transform electricity consumption data into meaningful insights through an interactive dashboard.

It helps users understand electricity consumption trends, identify unusual usage patterns, and estimate next-hour electricity consumption.

## 🎯 Problem Statement

Electricity consumption data is generated continuously, making it difficult to manually monitor usage patterns and identify unusual consumption behavior.

Traditional monitoring approaches may not provide sufficient insights into consumption trends or future demand.

This project addresses these challenges by developing a data analytics platform that uses Machine Learning techniques for anomaly detection and electricity demand forecasting.

## 🎯 Project Objectives

- Analyze household electricity consumption patterns.
- Perform Exploratory Data Analysis (EDA).
- Identify unusual electricity consumption patterns.
- Develop Machine Learning models for anomaly detection.
- Forecast next-hour electricity consumption.
- Compare the performance of multiple ML models.
- Track Machine Learning experiments using MLflow.
- Build an interactive dashboard using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming and data processing |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Anomaly classification |
| LightGBM | Electricity demand forecasting |
| SQLite | MLflow tracking backend |
| MLflow | Experiment tracking |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| Git & GitHub | Version control |

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Anomaly Detection Model Development
6. Electricity Demand Forecasting
7. Model Evaluation and Comparison
8. MLflow Experiment Tracking
9. Streamlit Dashboard Development
10. GitHub Version Control

## 🤖 Machine Learning Models

### 1. Anomaly Detection – Classification

The project uses the following classification algorithms:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Isolation Forest

**Best-performing classification model:** XGBoost, based on the evaluated F1 score.

### 2. Electricity Demand Forecasting – Regression

The following regression algorithms are evaluated:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

**Best-performing forecasting model:** LightGBM, based on the evaluated MAE.

## 📈 Model Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## 📊 Streamlit Dashboard Features

The project provides an interactive dashboard with five main sections.

### 1. Dashboard Overview
- Total electricity records
- Average active power
- Average voltage
- Electricity consumption trend
- Hourly consumption analysis

### 2. Anomaly Detection
- XGBoost-based anomaly predictions
- Normal and anomalous record counts
- Anomaly probability
- Anomaly distribution visualization
- Predicted anomaly records

### 3. Demand Forecasting
- LightGBM-based forecasting
- Predicted next-hour active power
- Actual versus predicted consumption chart
- Recent forecasting results

### 4. Model Comparison
- Classification model performance comparison
- Regression model performance comparison
- Interactive performance charts

### 5. Dataset Explorer
- Dataset preview
- Dataset dimensions
- Statistical summary
- Dataset download option

## 📁 Project Structure

```text
Smart_grid_uci/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── household_power_clean.csv
│       └── household_power_features.csv
│
├── models/
│   ├── artifacts/
│   ├── train_anomaly.py
│   ├── train_forecast.py
│   ├── model_evaluation.py
│   └── mlflow_tracking.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│
├── utils/
│   └── features.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sakilasaki31-afk/smart_grid.git
```

### Step 2: Navigate to the Project Folder

```bash
cd smart_grid
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Application

```bash
streamlit run app/app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

## 🔄 Model Training Instructions

Run the following commands from the project root directory.

**Feature Engineering**

```bash
python utils/features.py
```

**Anomaly Detection Training**

```bash
python models/train_anomaly.py
```

**Demand Forecasting Training**

```bash
python models/train_forecast.py
```

**Model Evaluation**

```bash
python models/model_evaluation.py
```

**MLflow Experiment Tracking**

```bash
python models/mlflow_tracking.py
```

To open the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://localhost:5000
```

## 📌 Expected Outcomes

- Interactive electricity consumption analytics dashboard.
- Identification of unusual consumption patterns.
- Machine Learning-based anomaly classification.
- Next-hour electricity consumption estimates.
- Performance comparison of multiple ML algorithms.
- MLflow-based experiment tracking.

## ⚠️ Important Notes

- Anomaly labels are generated using feature-engineering rules and are not verified electricity theft or equipment fault labels.
- Forecasting predictions depend on the feature-engineered dataset and its training configuration.
- The dashboard currently uses stored dataset records, not a live smart-meter sensor connection.
- The processed datasets and trained model artifacts must be available locally for the application to run.

## 🚀 Future Enhancements

- Real-time electricity consumption monitoring.
- Live smart-meter integration.
- Advanced time-series forecasting.
- Cloud deployment.
- Automated anomaly alerts.
- Enhanced energy consumption recommendations.

