from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def predict(
    classifier: RandomForestClassifier, inference_input: pd.DataFrame, features: list[str], model_version: str | int
) -> pd.DataFrame:
    """Score a batch. Each row records the registry version that made the prediction."""
    predictions = inference_input.copy()
    predictions["predicted_species"] = classifier.predict(inference_input[features])
    predictions["confidence"] = classifier.predict_proba(inference_input[features]).max(axis=1).round(4)
    predictions["model_version"] = str(model_version)
    return predictions
