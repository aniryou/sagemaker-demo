"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from iris_demo.pipelines import data_processing, data_science, inference


def register_pipelines() -> dict[str, Pipeline]:
    processing = data_processing.create_pipeline()
    science = data_science.create_pipeline()
    return {
        "data_processing": processing,
        "data_science": science,
        "training": science.only_nodes("train_model"),  # SageMaker Train step
        "evaluation": science.only_nodes("evaluate_model"),  # SageMaker Evaluate step
        "inference": inference.create_pipeline(),
        "__default__": processing + science,
    }
