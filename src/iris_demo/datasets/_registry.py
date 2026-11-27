"""Helpers for the SageMaker Model Registry that the registry datasets share."""

from __future__ import annotations

from typing import Any

from kedro.io import DatasetError


def group_arn(sagemaker: Any, group: str) -> str:
    return sagemaker.describe_model_package_group(ModelPackageGroupName=group)["ModelPackageGroupArn"]


def package_arn(group_arn_: str, version: str) -> str:
    """The ARN of one version: arn:aws:sagemaker:<region>:<account>:model-package/<group>/<version>."""
    return group_arn_.replace(":model-package-group/", ":model-package/") + "/" + version


def tags_of(sagemaker: Any, group_arn_: str) -> dict[str, str]:
    return {tag["Key"]: tag["Value"] for tag in sagemaker.list_tags(ResourceArn=group_arn_)["Tags"]}


def describe_tagged_version(sagemaker: Any, group: str, tag: str) -> tuple[str, dict]:
    """Return the version in the ``tag`` of the group and the description of that version.

    Raises:
        DatasetError: The group has no such tag.
    """
    arn = group_arn(sagemaker, group)
    version = tags_of(sagemaker, arn).get(tag)
    if version is None:
        raise DatasetError(
            f"Model package group '{group}' has no tag '{tag}'. Run the train pipeline first: "
            "it registers a model version and sets this tag."
        )
    return version, sagemaker.describe_model_package(ModelPackageName=package_arn(arn, version))
