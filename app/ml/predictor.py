from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import reading_to_vector
from .explainability import Explainer


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


STATUS_NAMES = {
    0: "NORMAL",
    1: "WARNING",
    2: "CRITICAL",
}


class Predictor:

    def __init__(self):

        self.classifier = joblib.load(
            MODEL_DIR / "failure_model.joblib"
        )

        self.anomaly_model = joblib.load(
            MODEL_DIR / "anomaly_model.joblib"
        )

        self.feature_names = joblib.load(
            MODEL_DIR / "feature_names.joblib"
        )

        self.explainer = Explainer(
            self.classifier,
            self.feature_names
        )

    # ---------------------------------------------------------
    # FAULT DIAGNOSIS
    # ---------------------------------------------------------

    def _faults(self, r):

        faults = []

        if (
            r["vibration"] >= 0.55
            and r["temperature"] >= 80
        ):
            faults.append(
                "Possible bearing degradation or mechanical friction"
            )

        elif r["vibration"] >= 0.55:

            faults.append(
                "Possible bearing or mechanical vibration problem"
            )

        elif r["temperature"] >= 80:

            faults.append(
                "Possible cooling-system or overheating problem"
            )

        if r["pressure"] >= 3.5:

            faults.append(
                "Abnormal pressure"
            )

        if r["rpm"] <= 1150:

            faults.append(
                "Abnormal RPM / possible mechanical resistance"
            )

        if r["current"] >= 11:

            faults.append(
                "High electrical current"
            )

        if not faults:

            faults.append(
                "No specific fault pattern detected"
            )

        return faults

    # ---------------------------------------------------------
    # MAINTENANCE ACTIONS
    # ---------------------------------------------------------

    def _actions(self, fault, status):

        actions = []

        fault_lower = fault.lower()

        if (
            "bearing" in fault_lower
            or "mechanical friction" in fault_lower
        ):

            actions.extend(
                [
                    "Inspect bearing assembly",
                    "Check lubrication and shaft alignment",
                    "Inspect abnormal vibration sources",
                ]
            )

        if (
            "overheating" in fault_lower
            or "cooling" in fault_lower
        ):

            actions.extend(
                [
                    "Check cooling system and ventilation",
                    "Inspect temperature sensor",
                    "Check machine load",
                ]
            )

        if "pressure" in fault_lower:

            actions.extend(
                [
                    "Inspect pressure system and valves",
                    "Check pressure sensor and connections",
                ]
            )

        if "rpm" in fault_lower:

            actions.extend(
                [
                    "Inspect mechanical resistance and rotating components",
                    "Check motor/drive operation",
                ]
            )

        if "current" in fault_lower:

            actions.extend(
                [
                    "Inspect electrical connections",
                    "Check motor load and current draw",
                ]
            )

        if not actions:

            actions = [
                "Continue scheduled preventive maintenance",
                "Keep monitoring sensor trends",
            ]

        if status == "CRITICAL":

            actions.insert(
                0,
                "Follow site safety procedure before continued operation"
            )

        return list(dict.fromkeys(actions))

    # ---------------------------------------------------------
    # MAINTENANCE PRIORITY
    # ---------------------------------------------------------

    def _priority(
        self,
        status,
        anomaly,
        severity
    ):

        if (
            status == "CRITICAL"
            or (
                anomaly
                and severity == "HIGH"
            )
        ):

            return "URGENT"

        if (
            status == "WARNING"
            or anomaly
        ):

            return "HIGH"

        return "LOW"

    # ---------------------------------------------------------
    # HEALTH SCORE
    # ---------------------------------------------------------

    def _health_score(self, r):

        temperature_risk = np.clip(
            (r["temperature"] - 55) / 50,
            0,
            1
        )

        vibration_risk = np.clip(
            (r["vibration"] - 0.15) / 0.70,
            0,
            1
        )

        pressure_risk = np.clip(
            (r["pressure"] - 2.2) / 2.0,
            0,
            1
        )

        rpm_risk = np.clip(
            (1400 - r["rpm"]) / 450,
            0,
            1
        )

        current_risk = np.clip(
            (r["current"] - 6) / 8,
            0,
            1
        )

        age_risk = np.clip(
            (r["operating_hours"] - 4000) / 11000,
            0,
            1
        )

        risk = (
            0.30 * temperature_risk
            + 0.32 * vibration_risk
            + 0.10 * pressure_risk
            + 0.10 * rpm_risk
            + 0.10 * current_risk
            + 0.08 * age_risk
        )

        return round(
            float((1 - risk) * 100),
            2
        )

    # ---------------------------------------------------------
    # CONTRIBUTING FACTORS
    # ---------------------------------------------------------

    def _contributors(self, r):

        checks = [

            (
                "temperature",
                r["temperature"] >= 80,
                f"Temperature is elevated ({r['temperature']:.1f})"
            ),

            (
                "vibration",
                r["vibration"] >= 0.55,
                f"Vibration is elevated ({r['vibration']:.2f})"
            ),

            (
                "pressure",
                r["pressure"] >= 3.5,
                f"Pressure is elevated ({r['pressure']:.2f})"
            ),

            (
                "rpm",
                r["rpm"] <= 1150,
                f"RPM is low ({r['rpm']:.0f})"
            ),

            (
                "current",
                r["current"] >= 11,
                f"Current is elevated ({r['current']:.1f})"
            ),

            (
                "operating_hours",
                r["operating_hours"] >= 11000,
                f"Operating hours are high ({r['operating_hours']:.0f})"
            ),
        ]

        return [
            message
            for _, condition, message in checks
            if condition
        ] or [
            "No major sensor threshold exceeded"
        ]

    # ---------------------------------------------------------
    # MAIN PREDICTION
    # ---------------------------------------------------------

    def predict(self, reading: dict):

        # Convert input into model features
        x = reading_to_vector(reading)

        x_df = pd.DataFrame(
            x,
            columns=self.feature_names
        )

        # ML prediction
        prediction = int(
            self.classifier.predict(x_df)[0]
        )

        probabilities = (
            self.classifier
            .predict_proba(x_df)[0]
        )

        classes = [
            int(c)
            for c in self.classifier.classes_
        ]

        # Status probabilities
        status_probabilities = {}

        for class_id, probability in zip(
            classes,
            probabilities
        ):

            status_probabilities[
                STATUS_NAMES[class_id].lower()
            ] = round(
                float(probability),
                4
            )

        status = STATUS_NAMES[prediction]

        predicted_probability = float(
            probabilities[
                classes.index(prediction)
            ]
        )

        # -----------------------------------------------------
        # SHAP EXPLANATION
        # -----------------------------------------------------

        explanations = self.explainer.explain(
            x_df,
            prediction
        )

        # -----------------------------------------------------
        # ANOMALY DETECTION
        # -----------------------------------------------------

        anomaly_raw = float(
            self.anomaly_model
            .decision_function(x_df)[0]
        )

        anomaly = bool(
            self.anomaly_model
            .predict(x_df)[0] == -1
        )

        if anomaly_raw < -0.15:

            anomaly_severity = "HIGH"

        elif anomaly_raw < 0:

            anomaly_severity = "MEDIUM"

        else:

            anomaly_severity = "LOW"

        # -----------------------------------------------------
        # FAULT + HEALTH
        # -----------------------------------------------------

        faults = self._faults(reading)

        likely_fault = faults[0]

        health_score = self._health_score(
            reading
        )

        maintenance_priority = self._priority(
            status,
            anomaly,
            anomaly_severity
        )

        recommended_actions = self._actions(
            likely_fault,
            status
        )

        # -----------------------------------------------------
        # RECOMMENDATION
        # -----------------------------------------------------

        if status == "NORMAL":

            recommendation = (
                "Continue normal monitoring and "
                "scheduled preventive maintenance."
            )

        elif status == "WARNING":

            recommendation = (
                "Increase monitoring and inspect "
                "the indicated components before "
                "the condition worsens."
            )

        else:

            recommendation = (
                "Follow safety procedure, inspect "
                "the machine before continued operation, "
                "and schedule preventive maintenance."
            )

        # -----------------------------------------------------
        # FINAL RESPONSE
        # -----------------------------------------------------

        return {

            "machine_status":
                status,

            "status_probability":
                round(
                    predicted_probability,
                    4
                ),

            "status_probabilities":
                status_probabilities,

            "health_score":
                health_score,

            "anomaly":
                anomaly,

            "anomaly_score":
                round(
                    anomaly_raw,
                    4
                ),

            "anomaly_severity":
                anomaly_severity,

            "likely_fault":
                likely_fault,

            "maintenance_priority":
                maintenance_priority,

            "recommendation":
                recommendation,

            "recommended_actions":
                recommended_actions,

            "contributing_factors":
                self._contributors(reading),

            "explanations":
                explanations,
        }