"""Tests of the Model Registry datasets with a stubbed SageMaker client. They need no AWS account."""

from datetime import datetime

import boto3
import fsspec
import joblib
import pytest
from botocore.stub import ANY, Stubber
from kedro.io import DatasetError

from iris_demo.datasets import SageMakerModelDataset, SageMakerModelMetricsDataset

GROUP = "iris-demo-models"
GROUP_ARN = f"arn:aws:sagemaker:us-west-2:123456789012:model-package-group/{GROUP}"
PACKAGE_ARN = f"arn:aws:sagemaker:us-west-2:123456789012:model-package/{GROUP}/3"
MODEL = {"kind": "fake model", "weights": [1, 2, 3]}


@pytest.fixture
def sagemaker():
    client = boto3.client("sagemaker", region_name="us-west-2", aws_access_key_id="test", aws_secret_access_key="test")
    with Stubber(client) as stubber:
        yield client, stubber
        stubber.assert_no_pending_responses()


def describe_group_response() -> dict:
    return {
        "ModelPackageGroupName": GROUP,
        "ModelPackageGroupArn": GROUP_ARN,
        "CreationTime": datetime(2026, 1, 1),
        "CreatedBy": {},
        "ModelPackageGroupStatus": "Completed",
    }


def describe_package_response(**extra) -> dict:
    return {
        "ModelPackageName": GROUP,
        "ModelPackageArn": PACKAGE_ARN,
        "ModelPackageVersion": 3,
        "ModelPackageGroupName": GROUP,
        "CreationTime": datetime(2026, 1, 1),
        "ModelPackageStatus": "Completed",
        "ModelApprovalStatus": "Approved",
        "CreatedBy": {},
        **extra,
    }


def stub_tag_lookup(stubber, tags: dict) -> None:
    stubber.add_response("describe_model_package_group", describe_group_response(), {"ModelPackageGroupName": GROUP})
    stubber.add_response("list_tags", {"Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}, {"ResourceArn": GROUP_ARN})


def test_save_registers_a_version_from_the_uploaded_file_and_moves_the_tag(sagemaker):
    client, stubber = sagemaker
    dataset = SageMakerModelDataset(model_package_group=GROUP, model_data_uri="memory://models/", save_args={"tag": "champion"})
    dataset._client = client
    stubber.add_response(
        "create_model_package",
        {"ModelPackageArn": PACKAGE_ARN},
        {"ModelPackageGroupName": GROUP, "SourceUri": ANY, "ModelApprovalStatus": "Approved"},
    )
    stubber.add_response("describe_model_package_group", describe_group_response(), {"ModelPackageGroupName": GROUP})
    stubber.add_response("add_tags", {"Tags": []}, {"ResourceArn": GROUP_ARN, "Tags": [{"Key": "champion", "Value": "3"}]})

    dataset.save(MODEL)

    files = fsspec.filesystem("memory").glob("models/*/model.joblib")
    assert len(files) == 1
    with fsspec.open(f"memory://{files[0]}", "rb") as file:
        assert joblib.load(file) == MODEL


def test_load_reads_the_source_of_the_tagged_version(sagemaker):
    client, stubber = sagemaker
    source_uri = "memory://models/20260101T000000Z/model.joblib"
    with fsspec.open(source_uri, "wb") as file:
        joblib.dump(MODEL, file)
    dataset = SageMakerModelDataset(model_package_group=GROUP, model_data_uri="memory://models")
    dataset._client = client
    stub_tag_lookup(stubber, {"champion": "3"})
    stubber.add_response("describe_model_package", describe_package_response(SourceUri=source_uri), {"ModelPackageName": PACKAGE_ARN})

    assert dataset.load() == MODEL


def test_load_without_the_tag_tells_the_user_to_train(sagemaker):
    client, stubber = sagemaker
    dataset = SageMakerModelDataset(model_package_group=GROUP, model_data_uri="memory://models")
    dataset._client = client
    stub_tag_lookup(stubber, {})

    with pytest.raises(DatasetError, match="no tag 'champion'"):
        dataset.load()


def test_exists_means_the_tag_is_set(sagemaker):
    client, stubber = sagemaker
    dataset = SageMakerModelDataset(model_package_group=GROUP, model_data_uri="memory://models", load_args={"tag": "challenger"})
    dataset._client = client
    stub_tag_lookup(stubber, {"champion": "3"})

    assert not dataset.exists()
    assert dataset._describe()["load_args"] == {"tag": "challenger"}
    assert dataset._describe()["save_args"] == {"tag": "champion", "approval_status": "Approved"}


def test_metrics_become_metadata_of_the_tagged_version(sagemaker):
    client, stubber = sagemaker
    dataset = SageMakerModelMetricsDataset(model_package_group=GROUP)
    dataset._client = client
    stub_tag_lookup(stubber, {"champion": "3"})
    stubber.add_response("describe_model_package", describe_package_response(), {"ModelPackageName": PACKAGE_ARN})
    stubber.add_response(
        "update_model_package",
        {"ModelPackageArn": PACKAGE_ARN},
        {
            "ModelPackageArn": PACKAGE_ARN,
            "CustomerMetadataProperties": {"accuracy": "0.9667", "test_rows": "30", "model.max_depth": "3"},
        },
    )

    dataset.save({"accuracy": 0.9667, "test_rows": 30, "model": {"max_depth": 3}})


def test_metrics_load_returns_the_metadata(sagemaker):
    client, stubber = sagemaker
    dataset = SageMakerModelMetricsDataset(model_package_group=GROUP)
    dataset._client = client
    stub_tag_lookup(stubber, {"champion": "3"})
    stubber.add_response(
        "describe_model_package",
        describe_package_response(CustomerMetadataProperties={"accuracy": "0.9667"}),
        {"ModelPackageName": PACKAGE_ARN},
    )

    assert dataset.load() == {"accuracy": "0.9667"}
