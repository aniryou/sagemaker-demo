from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

import fsspec
import pandas as pd
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class CSVDataset(AbstractDataset[pd.DataFrame, pd.DataFrame]):
    """Read and write a CSV file with pandas on any fsspec filesystem, for example S3.

    The project uses this dataset, not ``kedro_datasets.pandas.CSVDataset``. That
    dataset passes ``storage_options`` to pandas, which requires pandas 1.2. The
    SageMaker scikit-learn image has pandas 1.1.3. A newer pandas in that image
    causes errors in its compiled numpy and scikit-learn packages.

    Example catalog entry::

        iris_raw:
          type: iris_demo.datasets.CSVDataset
          filepath: s3://my-bucket/data/01_raw/iris.csv
          load_args:
            sep: ","
          save_args:
            index: false
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {"index": False}

    def __init__(
        self,
        *,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new instance of ``CSVDataset``.

        Args:
            filepath: Path to the CSV file, with a protocol prefix for a remote
                filesystem, for example ``s3://bucket/key.csv``.
            load_args: Arguments of ``pandas.read_csv``.
            save_args: Arguments of ``pandas.DataFrame.to_csv``. The default is
                ``index: False``.
            credentials: Credentials of the filesystem, passed to ``fsspec``.
            fs_args: Other arguments of the ``fsspec`` filesystem.
            metadata: Any arbitrary metadata. Kedro does not use it.
        """
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}
        protocol, path = get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)
        self.metadata = metadata
        self._load_args = {**self.DEFAULT_LOAD_ARGS, **(load_args or {})}
        self._save_args = {**self.DEFAULT_SAVE_ARGS, **(save_args or {})}

    def load(self) -> pd.DataFrame:
        with self._fs.open(self._path, mode="rb") as file:
            return pd.read_csv(file, **self._load_args)

    def save(self, data: pd.DataFrame) -> None:
        # ``to_csv`` returns text. Thus, the dataset writes the bytes itself. This
        # works with every pandas version.
        with self._fs.open(self._path, mode="wb") as file:
            file.write(data.to_csv(**self._save_args).encode("utf-8"))
        self._invalidate_cache()

    @property
    def _path(self) -> str:
        return get_filepath_str(self._filepath, self._protocol)

    def _exists(self) -> bool:
        return self._fs.exists(self._path)

    def _release(self) -> None:
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._fs.invalidate_cache(self._path)

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._path,
            "protocol": self._protocol,
            "load_args": self._load_args,
            "save_args": self._save_args,
        }
