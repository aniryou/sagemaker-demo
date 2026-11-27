from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(train_data: pd.DataFrame, features: list[str], target: str, model: dict) -> RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=model["n_estimators"],
        max_depth=model["max_depth"],
        random_state=model["random_state"],
    )
    classifier.fit(train_data[features], train_data[target])
    return classifier
