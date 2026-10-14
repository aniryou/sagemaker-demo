from kedro.pipeline import Pipeline, node

from .nodes import clean_iris, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                clean_iris,
                inputs=["iris_raw", "params:features", "params:target"],
                outputs="iris_clean",
                name="clean_iris",
            ),
            node(
                split_data,
                inputs=["iris_clean", "params:target", "params:split"],
                outputs=["train_data", "test_data"],
                name="split_data",
            ),
        ]
    )
