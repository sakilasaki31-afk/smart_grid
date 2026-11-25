
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


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

FEATURE_COLUMNS_FILE = (
    ARTIFACT_DIR / "anomaly_feature_columns.pkl"
)

FEATURE_MEDIANS_FILE = (
    ARTIFACT_DIR / "anomaly_feature_medians.pkl"
)


# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

def load_dataset():

    print("Loading feature-engineered dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Feature dataset not found. Run utils/features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("Dataset shape:", df.shape)

    return df


# ==========================================
# STEP 3: PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(df):

    target = "is_anomaly"

    if target not in df.columns:
        raise ValueError("is_anomaly column not found!")

    leakage_columns = [
        "datetime",
        "Global_active_power",
        "Unmetered",
        "Sub_metering_total",
        "power_zscore",
        "zero_reading",
        "statistical_outlier",
        "extreme_peak",
        "reactive_active_ratio",
        "submeter_ratio",
        "voltage_power_interaction",
        "is_anomaly",
        "anomaly_type",
        "target_next_hour"
    ]

    X = df.select_dtypes(
        include=[np.number]
    ).drop(
        columns=leakage_columns,
        errors="ignore"
    )

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Keep valid binary target records
    valid_rows = (
        y.notna() &
        np.isfinite(y) &
        y.isin([0, 1])
    )

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int).copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("Number of features:", X.shape[1])

    print("\nTarget distribution:")
    print(y.value_counts())

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

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both train and test datasets must contain "
            "normal and anomaly records."
        )

    # Calculate medians using training data only
    medians = X_train.median()

    # Handle columns containing only missing values
    medians = medians.fillna(0)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Save feature names and their order
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
    print("Features saved:", len(feature_columns))
    print("Feature columns and medians saved.")

    return X_train, X_test, y_train, y_test


# ==========================================
# STEP 5: EVALUATION FUNCTION
# ==========================================

results = []


def evaluate_model(name, y_true, y_pred, y_score):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_true, y_score)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    print("\n================================")
    print("MODEL:", name)
    print("================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


# ==========================================
# STEP 6: TRAIN CLASSIFICATION MODELS
# ==========================================

def train_classification_models(
    X_train, X_test, y_train, y_test
):

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(
                (y_train == 0).sum() /
                max((y_train == 1).sum(), 1)
            ),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    fitted_models = {}

    for name, model in models.items():

        print("\nTraining:", name)

        if name == "Gradient Boosting":

            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights
            )

        else:

            model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        evaluate_model(
            name,
            y_test,
            predictions,
            probabilities
        )

        fitted_models[name] = model

        model_filename = (
            name.lower().replace(" ", "_") + ".pkl"
        )

        joblib.dump(
            model,
            ARTIFACT_DIR / model_filename
        )

        print("Model saved:", model_filename)

    return fitted_models


# ==========================================
# STEP 7: ISOLATION FOREST
# ==========================================

def train_isolation_forest(
    X_train, X_test, y_train, y_test
):

    print("\nTraining: Isolation Forest")

    contamination = float(y_train.mean())

    contamination = min(
        max(contamination, 0.001),
        0.5
    )

    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    predictions_raw = model.predict(X_test)

    # -1 = anomaly, 1 = normal
    predictions = np.where(
        predictions_raw == -1, 1, 0
    )

    anomaly_scores = -model.decision_function(X_test)

    evaluate_model(
        "Isolation Forest",
        y_test,
        predictions,
        anomaly_scores
    )

    joblib.dump(
        model,
        ARTIFACT_DIR / "isolation_forest.pkl"
    )

    print("Isolation Forest model saved.")

    return model


# ==========================================
# STEP 8: SAVE RESULTS
# ==========================================

def save_results():

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False
    )

    output_file = (
        ARTIFACT_DIR / "anomaly_model_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n================================")
    print("MODEL COMPARISON RESULTS")
    print("================================")

    print(results_df.to_string(index=False))

    print("\nResults saved:", output_file)

    return results_df


# ==========================================
# STEP 9: MAIN EXECUTION
# ==========================================

def main():

    df = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    print("\nStarting classification training...")

    train_classification_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nStarting anomaly detection...")

    train_isolation_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_results()

    print("\n================================")
    print("ANOMALY DETECTION PIPELINE COMPLETED!")
    print("================================")

    print("\nDashboard preprocessing files saved:")
    print(FEATURE_COLUMNS_FILE)
    print(FEATURE_MEDIANS_FILE)


if __name__ == "__main__":
    main()
