from __future__ import annotations

from typing import Any

import pandas as pd
import shap


class Explainer:
    """
    Generate local SHAP feature contributions for the
    predicted machine-status class.
    """

    def __init__(self, classifier: Any, feature_names: list[str]):
        self.classifier = classifier
        self.feature_names = feature_names

        try:
            self._explainer = shap.TreeExplainer(classifier)
        except Exception:
            self._explainer = None

    def explain(
        self,
        x: pd.DataFrame,
        predicted_class: int,
        top_n: int = 5,
    ) -> list[dict]:

        if self._explainer is None:
            return []

        try:
            shap_values = self._explainer.shap_values(x)

            # Different SHAP versions return different structures.
            if isinstance(shap_values, list):

                # One array per class
                class_values = shap_values[predicted_class][0]

            else:
                arr = shap_values

                if getattr(arr, "ndim", 0) == 3:
                    # Shape:
                    # samples × features × classes
                    class_values = arr[0, :, predicted_class]

                elif getattr(arr, "ndim", 0) == 2:
                    class_values = arr[0]

                else:
                    class_values = arr

            row = x.iloc[0]

            items = []

            for name, contribution in zip(
                self.feature_names,
                class_values
            ):

                contribution = float(contribution)

                items.append(
                    {
                        "feature": name,
                        "value": float(row[name]),
                        "contribution": round(
                            contribution,
                            4
                        ),
                        "direction": (
                            "increases risk"
                            if contribution >= 0
                            else "reduces risk"
                        ),
                    }
                )

            items.sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True
            )

            return items[:top_n]

        except Exception:
            # Explainability should never break the prediction API.
            return []