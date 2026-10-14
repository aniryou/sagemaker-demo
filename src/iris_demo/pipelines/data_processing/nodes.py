from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_iris(iris_raw: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Keep the model columns and remove the rows that have empty values."""
    return iris_raw[[*features, target]].dropna().reset_index(drop=True)


def split_data(iris_clean: pd.DataFrame, target: str, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide the data into a train set and a test set with the same class mix.

    The seed is fixed. Thus, each model version gets a score on the same test
    rows, and you can compare the accuracy of the challenger and the champion.
    """
    train, test = train_test_split(
        iris_clean,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=iris_clean[target],
    )
    return train, test
