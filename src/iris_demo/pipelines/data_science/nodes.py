from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_model(train_data: pd.DataFrame, features: list[str], target: str, model: dict) -> RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=model["n_estimators"],
        max_depth=model["max_depth"],
        random_state=model["random_state"],
    )
    classifier.fit(train_data[features], train_data[target])
    return classifier


def evaluate_model(
    classifier: RandomForestClassifier, test_data: pd.DataFrame, features: list[str], target: str, model: dict
) -> dict:
    """Score on the held-out split.

    `multiclass_classification_metrics` follows the SageMaker Model Registry
    model-quality format. `summary` holds the same numbers as strings, because
    the registry only accepts string metadata on a model version.
    """
    predicted = classifier.predict(test_data[features])
    accuracy = float(accuracy_score(test_data[target], predicted))
    weighted_f1 = float(f1_score(test_data[target], predicted, average="weighted"))
    return {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "weighted_f1": {"value": weighted_f1, "standard_deviation": "NaN"},
        },
        "summary": {
            "accuracy": f"{accuracy:.4f}",
            "weighted_f1": f"{weighted_f1:.4f}",
            "n_estimators": str(model["n_estimators"]),
            "max_depth": str(model["max_depth"]),
            "test_rows": str(len(test_data)),
        },
    }
