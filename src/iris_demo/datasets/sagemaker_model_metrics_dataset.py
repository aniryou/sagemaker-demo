from __future__ import annotations

import logging
from typing import Any

import boto3
from kedro.io import AbstractDataset

from iris_demo.datasets import _registry

logger = logging.getLogger(__name__)


class SageMakerModelMetricsDataset(AbstractDataset[dict, dict]):
    """Store the metrics of a registered model version as metadata of that version.

    ``save`` writes a dictionary of metrics into the customer metadata of the
    version that the tag of the group points to. ``load`` returns that metadata.
    The Model Registry shows the metadata with the version in SageMaker Studio.

    The registry stores each value as a string. Nested dictionaries become keys
    with dots, for example ``model.max_depth``.

    Example catalog entry::

        evaluation:
          type: iris_demo.datasets.SageMakerModelMetricsDataset
          model_package_group: iris-demo-models
          save_args:
            tag: champion
          load_args:
            tag: champion
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {"tag": "champion"}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {"tag": "champion"}

    def __init__(
        self,
        *,
        model_package_group: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new instance of ``SageMakerModelMetricsDataset``.

        Args:
            model_package_group: Name of the model package group in the Model Registry.
            load_args: ``tag``: the group tag that contains the version to read. The default is ``champion``.
            save_args: ``tag``: the group tag that contains the version to update. The default is ``champion``.
            metadata: Any arbitrary metadata. Kedro does not use it.
        """
        self._group = model_package_group
        self._load_args = {**self.DEFAULT_LOAD_ARGS, **(load_args or {})}
        self._save_args = {**self.DEFAULT_SAVE_ARGS, **(save_args or {})}
        self.metadata = metadata
        self._client = None

    @property
    def _sagemaker(self) -> Any:
        if self._client is None:
            self._client = boto3.client("sagemaker")
        return self._client

    def load(self) -> dict:
        _, package = _registry.describe_tagged_version(self._sagemaker, self._group, self._load_args["tag"])
        return dict(package.get("CustomerMetadataProperties", {}))

    def save(self, data: dict) -> None:
        tag = self._save_args["tag"]
        version, package = _registry.describe_tagged_version(self._sagemaker, self._group, tag)
        properties = {key: str(value) for key, value in _flatten(data).items()}
        self._sagemaker.update_model_package(ModelPackageArn=package["ModelPackageArn"], CustomerMetadataProperties=properties)
        logger.info("Stored %d metrics on version %s of '%s' (tag '%s').", len(properties), version, self._group, tag)

    def _exists(self) -> bool:
        arn = _registry.group_arn(self._sagemaker, self._group)
        return self._load_args["tag"] in _registry.tags_of(self._sagemaker, arn)

    def _describe(self) -> dict[str, Any]:
        return {"model_package_group": self._group, "load_args": self._load_args, "save_args": self._save_args}


def _flatten(data: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
