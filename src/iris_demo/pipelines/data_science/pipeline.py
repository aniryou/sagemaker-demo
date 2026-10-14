from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                train_model,
                inputs=["train_data", "params:features", "params:target", "params:model"],
                outputs="model",
                name="train_model",
            ),
            node(
                evaluate_model,
                inputs=["model", "test_data", "params:features", "params:target", "params:model"],
                outputs="evaluation",
                name="evaluate_model",
            ),
        ]
    )
