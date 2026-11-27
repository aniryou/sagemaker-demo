"""Register the project pipelines.

`train`, `evaluate` and `inference` are the pipelines that the SageMaker pipelines
run. Each SageMaker pipeline runs `kedro run --env sagemaker --pipeline <name>`.
"""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines()  # data_processing, training, evaluation, inference
    pipelines["train"] = pipelines["data_processing"] + pipelines["training"]
    pipelines["evaluate"] = pipelines["data_processing"] + pipelines["evaluation"]
    pipelines["__default__"] = pipelines["train"] + pipelines["evaluation"]
    return pipelines
