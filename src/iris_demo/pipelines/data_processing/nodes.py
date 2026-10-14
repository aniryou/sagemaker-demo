from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_iris(iris_raw: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Keep the modelling columns and drop incomplete rows."""
    return iris_raw[[*features, target]].dropna().reset_index(drop=True)


def split_data(iris_clean: pd.DataFrame, target: str, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split.

    The seed is fixed so every model version is scored on the same test rows,
    which is what makes challenger and champion accuracy comparable.
    """
    train, test = train_test_split(
        iris_clean,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=iris_clean[target],
    )
    return train, test
