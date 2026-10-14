from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from kedro.io import AbstractDataset


class CSVDataset(AbstractDataset):
    """Read and write a local CSV file with pandas.

    The project uses this dataset, not kedro_datasets.pandas.CSVDataset. That
    dataset requires pandas 1.2 or higher. The SageMaker scikit-learn 1.2-1 image
    has pandas 1.1.3. A newer pandas in that image causes errors in its numpy and
    scikit-learn binaries.
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
