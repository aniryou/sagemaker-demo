"""Shared resource names and AWS session for the deploy scripts."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import boto3

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "iris-demo"
_AWS_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION")

# The Kedro pipelines that the SageMaker pipelines run. The SageMaker pipeline of
# `train` is `iris-demo-train`, and so on.
PIPELINES = ("train", "evaluate", "inference")

# This tag on the model package group contains the version that evaluate and
# inference use: champion=<version>. The Kedro datasets set and read it.
CHAMPION = "champion"


def load_aws_env() -> None:
    """Export only the AWS_* entries of .env. The file can also contain other secrets.

    If the environment already has credentials or a profile, this function does
    not read .env. Thus, `AWS_PROFILE=other python deploy/pipelines.py ...` works.
    """
    env_file = ROOT / ".env"
    if not env_file.exists() or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        return
    for line in env_file.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in _AWS_KEYS:
            os.environ[key.strip()] = value.strip().strip("'\"")


def get_session() -> boto3.Session:
    load_aws_env()
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return boto3.Session(region_name=region)


def get_names(session: boto3.Session) -> SimpleNamespace:
    account = session.client("sts").get_caller_identity()["Account"]
    region = session.region_name
    group = f"{PREFIX}-models"
    return SimpleNamespace(
        account=account,
        region=region,
        bucket=f"{PREFIX}-sagemaker-{account}-{region}",
        model_group=group,
        model_group_arn=f"arn:aws:sagemaker:{region}:{account}:model-package-group/{group}",
        model_package_arn_prefix=f"arn:aws:sagemaker:{region}:{account}:model-package/{group}",
        pipelines={name: f"{PREFIX}-{name}" for name in PIPELINES},
        ecr_repository=PREFIX,
        ecr_repository_arn=f"arn:aws:ecr:{region}:{account}:repository/{PREFIX}",
        image_repository=f"{account}.dkr.ecr.{region}.amazonaws.com/{PREFIX}",
        execution_role=f"{PREFIX}-sagemaker-execution",
        studio_role=f"{PREFIX}-studio-user",
        studio_domain=f"{PREFIX}-studio",
        studio_user="default-user",
    )


def role_arn(names: SimpleNamespace, role_name: str) -> str:
    return f"arn:aws:iam::{names.account}:role/{role_name}"
