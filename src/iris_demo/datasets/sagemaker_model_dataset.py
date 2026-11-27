from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import boto3
import fsspec
import joblib
from kedro.io import AbstractDataset, DatasetError

from iris_demo.datasets import _registry

logger = logging.getLogger(__name__)


class SageMakerModelDataset(AbstractDataset[Any, Any]):
    """Register a model in the SageMaker Model Registry and load it back by tag.

    ``save`` writes the model with joblib to ``<model_data_uri>/<timestamp>/model.joblib``,
    registers a new version in the model package group with that file as its
    source, and points the tag of the group at the new version. ``load`` reads
    the version in the tag, downloads its source and loads it with joblib.

    SageMaker cannot put a tag on one model version. Thus, the dataset keeps the
    tags on the model package group, in the form ``<tag>=<version>``. The group
    has one value for each tag. Thus, only one champion can exist.

    The registry entry has no inference container. Kedro loads the model and runs
    the inference. Thus, the version has only a source URI. To deploy a version on
    a SageMaker endpoint, add an inference specification to it.

    Example catalog entry::

        model:
          type: iris_demo.datasets.SageMakerModelDataset
          model_package_group: iris-demo-models
          model_data_uri: s3://my-bucket/models
          save_args:
            tag: champion
            approval_status: Approved
          load_args:
            tag: champion

    The registry calls use the default boto3 credentials and region. The
    ``credentials`` and ``fs_args`` go to fsspec, which reads and writes the
    model file.
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {"tag": "champion"}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {"tag": "champion", "approval_status": "Approved"}

    def __init__(
        self,
        *,
        model_package_group: str,
        model_data_uri: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new instance of ``SageMakerModelDataset``.

        Args:
            model_package_group: Name of the model package group in the Model Registry.
            model_data_uri: Folder for the model files, for example ``s3://bucket/models``.
                Each ``save`` makes a new subfolder with a timestamp.
            load_args: ``tag``: the group tag that contains the version to load.
                The default is ``champion``.
            save_args: ``tag``: the group tag that receives the new version. The
                default is ``champion``. ``approval_status``: the status of the new
                version, ``Approved`` (the default), ``PendingManualApproval`` or
                ``Rejected``. ``description``: an optional description of the version.
            credentials: Credentials of the filesystem of ``model_data_uri``, passed to fsspec.
            fs_args: Other arguments of the fsspec filesystem.
            metadata: Any arbitrary metadata. Kedro does not use it.
        """
        self._group = model_package_group
        self._model_data_uri = model_data_uri.rstrip("/")
        self._load_args = {**self.DEFAULT_LOAD_ARGS, **(load_args or {})}
        self._save_args = {**self.DEFAULT_SAVE_ARGS, **(save_args or {})}
        self._fs_args = {**(deepcopy(credentials) or {}), **(deepcopy(fs_args) or {})}
        self.metadata = metadata
        self._client = None

    @property
    def _sagemaker(self) -> Any:
        if self._client is None:
            self._client = boto3.client("sagemaker")
        return self._client

    def load(self) -> Any:
        tag = self._load_args["tag"]
        version, package = _registry.describe_tagged_version(self._sagemaker, self._group, tag)
        source_uri = package.get("SourceUri")
        if not source_uri:
            raise DatasetError(f"Version {version} of '{self._group}' has no source URI. This dataset did not register it.")
        logger.info("Loading tag '%s' of '%s': version %s from %s", tag, self._group, version, source_uri)
        with fsspec.open(source_uri, mode="rb", **self._fs_args) as file:
            return joblib.load(file)

    def save(self, data: Any) -> None:
        tag = self._save_args["tag"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        source_uri = f"{self._model_data_uri}/{stamp}/model.joblib"
        with fsspec.open(source_uri, mode="wb", **self._fs_args) as file:
            joblib.dump(data, file)

        request = {
            "ModelPackageGroupName": self._group,
            "SourceUri": source_uri,
            "ModelApprovalStatus": self._save_args["approval_status"],
        }
        if self._save_args.get("description"):
            request["ModelPackageDescription"] = self._save_args["description"]
        arn = self._sagemaker.create_model_package(**request)["ModelPackageArn"]
        version = arn.rsplit("/", 1)[1]
        self._sagemaker.add_tags(
            ResourceArn=_registry.group_arn(self._sagemaker, self._group),
            Tags=[{"Key": tag, "Value": version}],
        )
        logger.info("Registered version %s of '%s' from %s. Tag '%s' now points to it.", version, self._group, source_uri, tag)

    def _exists(self) -> bool:
        arn = _registry.group_arn(self._sagemaker, self._group)
        return self._load_args["tag"] in _registry.tags_of(self._sagemaker, arn)

    def _describe(self) -> dict[str, Any]:
        return {
            "model_package_group": self._group,
            "model_data_uri": self._model_data_uri,
            "load_args": self._load_args,
            "save_args": self._save_args,
        }
