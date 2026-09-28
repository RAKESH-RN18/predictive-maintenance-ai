import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"


st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏭 AI-Powered Predictive Maintenance"
)

st.caption(
    "Machine health classification • "
    "anomaly detection • fault diagnosis • "
    "explainable AI • maintenance recommendations • "
    "real-world Scania benchmark"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Sensor Input")

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=150.0,
        value=72.0,
        step=0.5
    )

    vibration = st.number_input(
        "Vibration",
        min_value=0.0,
        max_value=10.0,
        value=0.40,
        step=0.01
    )

    pressure = st.number_input(
        "Pressure",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.1
    )

    rpm = st.number_input(
        "RPM",
        min_value=0.0,
        max_value=10000.0,
        value=1300.0,
        step=10.0
    )

    voltage = st.number_input(
        "Voltage",
        min_value=0.0,
        max_value=1000.0,
        value=225.0,
        step=1.0
    )

    current = st.number_input(
        "Current",
        min_value=0.0,
        max_value=100.0,
        value=9.0,
        step=0.5
    )

    operating_hours = st.number_input(
        "Operating Hours",
        min_value=0.0,
        max_value=200000.0,
        value=8000.0,
        step=100.0
    )

    run_diagnosis = st.button(
        "Run AI Diagnosis",
        type="primary",
        use_container_width=True
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if run_diagnosis:

    payload = {
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "voltage": voltage,
        "current": current,
        "operating_hours": operating_hours,
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        st.session_state["result"] = response.json()

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the API server is running."
        )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )


result = st.session_state.get("result")


# ============================================================
# RESULTS
# ============================================================

if result:

    status = result["machine_status"]

    st.markdown(
        f"## Machine Status: **{status}**"
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Health Score",
        f"{result['health_score']:.1f}%"
    )

    c2.metric(
        "Status Probability",
        f"{result['status_probability'] * 100:.1f}%"
    )

    c3.metric(
        "Anomaly",
        "YES" if result["anomaly"] else "NO"
    )

    c4.metric(
        "Maintenance Priority",
        result["maintenance_priority"]
    )

    st.divider()

    # ========================================================
    # STATUS + SENSOR
    # ========================================================

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### 📊 Status Probabilities"
        )

        probabilities = result[
            "status_probabilities"
        ]

        probability_df = pd.DataFrame(
            {
                "Status": [
                    "Normal",
                    "Warning",
                    "Critical"
                ],
                "Probability": [
                    probabilities.get(
                        "normal",
                        0
                    ),
                    probabilities.get(
                        "warning",
                        0
                    ),
                    probabilities.get(
                        "critical",
                        0
                    ),
                ],
            }
        )

        fig = px.bar(
            probability_df,
            x="Status",
            y="Probability",
            range_y=[0, 1],
            text_auto=".1%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            "### 📡 Sensor Snapshot"
        )

        sensor_df = pd.DataFrame(
            {
                "Sensor": [
                    "Temperature",
                    "Vibration",
                    "Pressure",
                    "RPM",
                    "Voltage",
                    "Current",
                    "Operating Hours",
                ],
                "Value": [
                    temperature,
                    vibration,
                    pressure,
                    rpm,
                    voltage,
                    current,
                    operating_hours,
                ],
            }
        )

        st.dataframe(
            sensor_df,
            hide_index=True,
            use_container_width=True
        )

    # ========================================================
    # ANOMALY
    # ========================================================

    st.markdown(
        "### 🚨 Anomaly Detection"
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Detected",
        "YES" if result["anomaly"] else "NO"
    )

    a2.metric(
        "Anomaly Score",
        f"{result['anomaly_score']:.4f}"
    )

    a3.metric(
        "Severity",
        result["anomaly_severity"]
    )

    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.markdown(
        "### 🔍 Why did the AI make this prediction?"
    )

    st.caption(
        "SHAP shows which sensor features contributed "
        "most to the predicted machine-status class."
    )

    explanations = pd.DataFrame(
        result.get(
            "explanations",
            []
        )
    )

    if not explanations.empty:

        explanations["Feature"] = (
            explanations["feature"]
            .str.replace(
                "_",
                " ",
                regex=False
            )
            .str.title()
        )

        chart_df = (
            explanations[
                [
                    "Feature",
                    "contribution"
                ]
            ]
            .copy()
            .sort_values(
                "contribution"
            )
        )

        fig = px.bar(
            chart_df,
            x="contribution",
            y="Feature",
            orientation="h",
            title="Top SHAP Feature Contributions"
        )

        fig.add_vline(
            x=0,
            line_width=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        display_df = explanations[
            [
                "Feature",
                "value",
                "contribution",
                "direction",
            ]
        ].rename(
            columns={
                "value": "Sensor Value",
                "contribution": "SHAP Contribution",
                "direction": "Effect",
            }
        )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "SHAP explanation is unavailable "
            "for this prediction."
        )

    # ========================================================
    # FAULT
    # ========================================================

    st.markdown(
        "### 🔧 Likely Fault"
    )

    st.info(
        result["likely_fault"]
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.markdown(
        "### 🛠️ Recommendation"
    )

    st.warning(
        result["recommendation"]
    )

    # ========================================================
    # RECOMMENDED ACTIONS
    # ========================================================

    st.markdown(
        "### 🔧 Recommended Actions"
    )

    for action in result[
        "recommended_actions"
    ]:

        st.write(
            f"• {action}"
        )

    # ========================================================
    # CONTRIBUTING FACTORS
    # ========================================================

    st.markdown(
        "### 📌 Contributing Factors"
    )

    for factor in result[
        "contributing_factors"
    ]:

        st.write(
            f"• {factor}"
        )


else:

    st.info(
        "Enter sensor values in the sidebar "
        "and click **Run AI Diagnosis**."
    )


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.subheader(
    "📜 Prediction History"
)

try:

    history_response = requests.get(
        f"{API_URL}/history?limit=30",
        timeout=5
    )

    history_response.raise_for_status()

    rows = history_response.json()

    if rows:

        history_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            history_df,
            hide_index=True,
            use_container_width=True
        )

        if "health_score" in history_df.columns:

            chart_df = (
                history_df
                .sort_values("created_at")
            )

            fig = px.line(
                chart_df,
                x="created_at",
                y="health_score",
                markers=True,
                title="Machine Health Score Trend"
            )

            fig.update_yaxes(
                range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.caption(
            "No predictions recorded yet."
        )

except Exception:

    st.caption(
        "Prediction history is unavailable "
        "until the FastAPI server is running."
    )


# ============================================================
# REAL SCANIA DATASET
# ============================================================

st.divider()

st.header(
    "🚛 Real-World Scania Dataset Evaluation"
)

st.caption(
    "A separate benchmark trained on the official "
    "UCI APS Failure at Scania Trucks dataset. "
    "This model is independent of the sensor dashboard above."
)


# ============================================================
# LOAD SCANIA RESULTS
# ============================================================

RESULTS_FILE = (
    MODELS_DIR /
    "scania_model_results.json"
)

COMPARISON_FILE = (
    MODELS_DIR /
    "scania_model_comparison.csv"
)

IMPORTANCE_FILE = (
    MODELS_DIR /
    "scania_feature_importance.csv"
)

CLEAN_DATA_FILE = (
    REAL_DATA_DIR /
    "scania_clean.csv"
)


if not RESULTS_FILE.exists():

    st.warning(
        "Scania model results were not found. "
        "Run `python scripts\\train_real_dataset.py` first."
    )

else:

    try:

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            scania_results = json.load(file)

        # ----------------------------------------------------
        # Helper to safely find values
        # ----------------------------------------------------

        def get_value(data, *keys, default=None):

            current = data

            for key in keys:

                if isinstance(current, dict):
                    current = current.get(key)

                else:
                    return default

            return (
                current
                if current is not None
                else default
            )

        best_model = (
            scania_results.get(
                "best_model",
                "XGBoost"
            )
        )

        threshold = (
            scania_results.get(
                "selected_threshold",
                0.07
            )
        )

        # ----------------------------------------------------
        # TOP INFORMATION
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Real Dataset Model"
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Selected Model",
            str(best_model)
        )

        r2.metric(
            "Failure Threshold",
            f"{float(threshold):.2f}"
        )

        r3.metric(
            "Training Dataset",
            "60,000"
        )

        r4.metric(
            "Features",
            "170"
        )

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Model Comparison"
        )

        if COMPARISON_FILE.exists():

            comparison_df = pd.read_csv(
                COMPARISON_FILE
            )

            # Normalize column names for display
            comparison_display = (
                comparison_df.copy()
            )

            comparison_display.columns = [
                str(column)
                .replace("_", " ")
                .title()
                for column in comparison_display.columns
            ]

            st.dataframe(
                comparison_display,
                hide_index=True,
                use_container_width=True
            )

            # -----------------------------------------------
            # F1 comparison
            # -----------------------------------------------

            f1_column = next(
                (
                    column
                    for column in comparison_df.columns
                    if column.lower() == "f1"
                ),
                None
            )

            model_column = next(
                (
                    column
                    for column in comparison_df.columns
                    if "model" in column.lower()
                ),
                None
            )

            if (
                f1_column is not None
                and model_column is not None
            ):

                chart_df = comparison_df[
                    [
                        model_column,
                        f1_column
                    ]
                ].copy()

                chart_df.columns = [
                    "Model",
                    "F1 Score"
                ]

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y="F1 Score",
                    range_y=[0, 1],
                    text_auto=".3f",
                    title="F1 Score Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                "Model comparison file is unavailable."
            )

        # ----------------------------------------------------
        # BEST MODEL METRICS
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Test Performance"
        )

        best_metrics = None

        # Different possible JSON structures are handled
        # so the dashboard remains compatible with the
        # training script.

        if isinstance(
            scania_results.get("models"),
            dict
        ):

            best_metrics = (
                scania_results["models"]
                .get(best_model)
            )

        if best_metrics is None:

            if isinstance(
                scania_results.get("results"),
                dict
            ):

                best_metrics = (
                    scania_results["results"]
                    .get(best_model)
                )

        if best_metrics is None:

            best_metrics = scania_results

        if isinstance(
            best_metrics,
            dict
        ):

            metric_candidates = {
                "Accuracy": [
                    "accuracy"
                ],
                "Precision": [
                    "precision"
                ],
                "Recall": [
                    "recall"
                ],
                "F1 Score": [
                    "f1",
                    "f1_score"
                ],
                "ROC-AUC": [
                    "roc_auc",
                    "roc_auc_score"
                ],
                "PR-AUC": [
                    "pr_auc",
                    "average_precision"
                ],
                "Test Cost": [
                    "test_cost",
                    "cost"
                ],
            }

            metric_values = {}

            for display_name, keys in metric_candidates.items():

                value = None

                for key in keys:

                    if key in best_metrics:

                        value = best_metrics[key]
                        break

                if value is not None:

                    metric_values[
                        display_name
                    ] = value

            if metric_values:

                metric_columns = st.columns(
                    min(
                        len(metric_values),
                        4
                    )
                )

                for index, (
                    name,
                    value
                ) in enumerate(
                    metric_values.items()
                ):

                    column = metric_columns[
                        index % len(metric_columns)
                    ]

                    if name == "Test Cost":

                        column.metric(
                            name,
                            f"{float(value):,.0f}"
                        )

                    else:

                        column.metric(
                            name,
                            f"{float(value) * 100:.2f}%"
                        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.markdown(
            "### 🎯 Confusion Matrix"
        )

        confusion_matrix = None

        if isinstance(
            best_metrics,
            dict
        ):

            confusion_matrix = (
                best_metrics.get(
                    "confusion_matrix"
                )
            )

        if confusion_matrix is None:

            confusion_matrix = (
                scania_results.get(
                    "confusion_matrix"
                )
            )

        if (
            confusion_matrix is not None
            and len(confusion_matrix) == 2
        ):

            cm = pd.DataFrame(
                confusion_matrix,
                index=[
                    "Actual Negative",
                    "Actual Positive"
                ],
                columns=[
                    "Predicted Negative",
                    "Predicted Positive"
                ]
            )

            st.dataframe(
                cm,
                use_container_width=True
            )

        else:

            st.info(
                "Confusion-matrix values are not stored "
                "in the saved results file."
            )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.markdown(
            "### 🔎 Real Dataset Feature Importance"
        )

        st.caption(
            "The Scania dataset uses anonymized feature "
            "names. These are model-derived importance "
            "values and should not be interpreted as "
            "physical sensor names."
        )

        if IMPORTANCE_FILE.exists():

            importance_df = pd.read_csv(
                IMPORTANCE_FILE
            )

            # Detect feature column
            feature_column = next(
                (
                    column
                    for column in importance_df.columns
                    if "feature" in column.lower()
                ),
                importance_df.columns[0]
            )

            importance_column = next(
                (
                    column
                    for column in importance_df.columns
                    if (
                        "importance" in column.lower()
                        or "gain" in column.lower()
                    )
                ),
                importance_df.columns[-1]
            )

            importance_display = (
                importance_df[
                    [
                        feature_column,
                        importance_column
                    ]
                ]
                .copy()
            )

            importance_display.columns = [
                "Feature",
                "Importance"
            ]

            importance_display[
                "Importance"
            ] = pd.to_numeric(
                importance_display[
                    "Importance"
                ],
                errors="coerce"
            )

            importance_display = (
                importance_display
                .dropna()
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(15)
            )

            fig = px.bar(
                importance_display.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Important Features"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance_display,
                hide_index=True,
                use_container_width=True
            )

        else:

            st.info(
                "Feature-importance file is unavailable."
            )

        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        st.markdown(
            "### 🗃️ Real Dataset Information"
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Total Samples",
            "60,000"
        )

        d2.metric(
            "Negative",
            "59,000"
        )

        d3.metric(
            "Positive",
            "1,000"
        )

        d4.metric(
            "Positive Rate",
            "1.67%"
        )

        st.info(
            "The Scania dataset is highly imbalanced. "
            "For this reason, accuracy alone is not a "
            "sufficient measure of model quality. "
            "Recall, PR-AUC and the dataset's cost-sensitive "
            "evaluation are also shown."
        )

        # ----------------------------------------------------
        # IMPORTANT PROJECT NOTE
        # ----------------------------------------------------

        with st.expander(
            "ℹ️ How this real dataset relates to this project"
        ):

            st.write(
                """
                The main dashboard above uses the project's
                sensor-based 3-class predictive-maintenance
                model:

                NORMAL → WARNING → CRITICAL

                The Scania model is a separate binary
                real-world benchmark trained on the UCI APS
                Failure at Scania Trucks dataset.

                The Scania dataset contains anonymized machine
                features, so its columns are not presented as
                temperature, vibration, pressure, RPM, voltage
                or current.

                Therefore, the Scania model is used as a
                real-world ML validation/benchmark component
                and does not replace the existing sensor
                prediction model.
                """
            )

    except Exception as error:

        st.error(
            f"Unable to load Scania evaluation results: {error}"
        )

        st.exception(error)