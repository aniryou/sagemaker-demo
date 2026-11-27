import pandas as pd

from iris_demo.datasets import CSVDataset

FRAME = pd.DataFrame({"sepal_length": [5.1, 4.9], "species": ["setosa", "setosa"]})


def test_local_round_trip_makes_the_parent_folders(tmp_path):
    dataset = CSVDataset(filepath=str(tmp_path / "05_model_input" / "train.csv"))
    assert not dataset.exists()
    dataset.save(FRAME)
    assert dataset.exists()
    pd.testing.assert_frame_equal(dataset.load(), FRAME)


def test_remote_filesystem_round_trip():
    # fsspec resolves the protocol. On SageMaker the protocol is s3.
    dataset = CSVDataset(filepath="memory://bucket/data/01_raw/iris.csv", save_args={"index": False})
    dataset.save(FRAME)
    pd.testing.assert_frame_equal(dataset.load(), FRAME)
    assert dataset._describe() == {
        "filepath": "memory://bucket/data/01_raw/iris.csv",
        "protocol": "memory",
        "load_args": {},
        "save_args": {"index": False},
    }
