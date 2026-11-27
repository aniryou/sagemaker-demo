from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(classifier: RandomForestClassifier, test_data: pd.DataFrame, features: list[str], target: str) -> dict:
    """Calculate the model score on the test split.

    The split has a fixed seed. Thus, each model version gets a score on the same
    rows. On SageMaker, the result becomes metadata of the model version in the
    Model Registry.
    """
    predicted = classifier.predict(test_data[features])
    params = classifier.get_params()
    return {
        "accuracy": round(float(accuracy_score(test_data[target], predicted)), 4),
        "weighted_f1": round(float(f1_score(test_data[target], predicted, average="weighted")), 4),
        "test_rows": int(len(test_data)),
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
    }
