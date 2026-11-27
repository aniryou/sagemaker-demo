from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def predict(classifier: RandomForestClassifier, inference_input: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Score a batch of rows."""
    predictions = inference_input.copy()
    predictions["predicted_species"] = classifier.predict(inference_input[features])
    predictions["confidence"] = classifier.predict_proba(inference_input[features]).max(axis=1).round(4)
    return predictions
