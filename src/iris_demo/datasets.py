from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from kedro.io import AbstractDataset


class CSVDataset(AbstractDataset):
    """Local CSV file read and written with plain pandas.

    Used instead of kedro_datasets.pandas.CSVDataset, which needs pandas >= 1.2,
    while the SageMaker scikit-learn 1.2-1 image ships pandas 1.1.3. Upgrading
    pandas in that image breaks its numpy/scikit-learn binaries.
    """

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath)

    def save(self, data: pd.DataFrame) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self._filepath, index=False)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}
