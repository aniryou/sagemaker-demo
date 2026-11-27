"""Tests of the pipeline registry and the Kedro environments."""

from pathlib import Path

import pytest
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

ROOT = Path(__file__).resolve().parents[1]


def node_names(pipeline) -> list[str]:
    return sorted(node.name for node in pipeline.nodes)


def test_each_sagemaker_pipeline_has_a_kedro_pipeline():
    configure_project("iris_demo")
    from iris_demo.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert node_names(pipelines["train"]) == ["clean_iris", "split_data", "train_model"]
    assert node_names(pipelines["evaluate"]) == ["clean_iris", "evaluate_model", "split_data"]
    assert node_names(pipelines["inference"]) == ["predict"]
    assert node_names(pipelines["__default__"]) == ["clean_iris", "evaluate_model", "split_data", "train_model"]


def test_sagemaker_environment_uses_s3_and_the_model_registry(monkeypatch):
    monkeypatch.setenv("IRIS_BUCKET", "my-bucket")
    bootstrap_project(ROOT)
    with KedroSession.create(project_path=ROOT, env="sagemaker") as session:
        catalog = session.load_context().catalog
        iris_raw = catalog.get("iris_raw")._describe()
        assert (iris_raw["protocol"], iris_raw["filepath"]) == ("s3", "my-bucket/data/01_raw/iris.csv")
        assert catalog.get("predictions")._describe()["filepath"] == "my-bucket/data/07_model_output/predictions.csv"
        model = catalog.get("model")._describe()
        assert model["model_package_group"] == "iris-demo-models"
        assert model["model_data_uri"] == "s3://my-bucket/models"
        assert model["save_args"] == {"tag": "champion", "approval_status": "Approved"}
        assert model["load_args"] == {"tag": "champion"}
        assert catalog.get("evaluation")._describe()["model_package_group"] == "iris-demo-models"


def test_sagemaker_environment_needs_the_bucket_variable(monkeypatch):
    monkeypatch.delenv("IRIS_BUCKET", raising=False)
    bootstrap_project(ROOT)
    with KedroSession.create(project_path=ROOT, env="sagemaker") as session:
        with pytest.raises(Exception, match="IRIS_BUCKET"):
            session.load_context().catalog.get("iris_raw")
