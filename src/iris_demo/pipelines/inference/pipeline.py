from kedro.pipeline import Pipeline, node

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                predict,
                inputs=["model", "inference_input", "params:features", "params:model_version"],
                outputs="predictions",
                name="predict",
            ),
        ]
    )
