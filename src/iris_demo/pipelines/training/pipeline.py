from kedro.pipeline import Pipeline, node

from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                train_model,
                inputs=["train_data", "params:features", "params:target", "params:model"],
                outputs="model",
                name="train_model",
            ),
        ]
    )
