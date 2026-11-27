from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                evaluate_model,
                inputs=["model", "test_data", "params:features", "params:target"],
                outputs="evaluation",
                name="evaluate_model",
            ),
        ]
    )
