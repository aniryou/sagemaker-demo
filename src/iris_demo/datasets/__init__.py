"""Custom Kedro datasets of the project: the SageMaker Model Registry."""

from iris_demo.datasets.sagemaker_model_dataset import SageMakerModelDataset
from iris_demo.datasets.sagemaker_model_metrics_dataset import SageMakerModelMetricsDataset

__all__ = ["SageMakerModelDataset", "SageMakerModelMetricsDataset"]
