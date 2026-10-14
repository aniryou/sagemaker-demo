"""Define, deploy and run the SageMaker train and inference pipelines.

    python deploy/pipelines.py deploy
    python deploy/pipelines.py run train -p ModelParams=model.n_estimators=200,model.max_depth=4
    python deploy/pipelines.py run inference
    python deploy/pipelines.py run inference -p ModelAlias=challenger   # shadow test of the challenger

Each compute step runs `kedro run --pipeline ...`. Train is a Training job
(train_entry.py). The other steps are Processing jobs (entrypoint.py). Thus, all
ML logic stays in the Kedro project.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tarfile
import tempfile
import time

from pathlib import Path

from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.lambda_step import LambdaOutput, LambdaOutputTypeEnum, LambdaStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from common import CHAMPION, PREFIX, ROOT, get_names, get_session, role_arn

SKLEARN_VERSION = "1.2-1"
INSTANCE_TYPE = "ml.t3.medium"  # Processing steps. The account has Processing quota only for ml.t3.
TRAINING_INSTANCE_TYPE = "ml.m5.large"
DATA = "/opt/ml/processing/data"  # must be the same as conf/sagemaker/globals.yml
PROJECT_FILES = ("pyproject.toml", "requirements.txt", "conf/base", "conf/sagemaker", "src/iris_demo")


def package_project(session, names) -> str:
    """Upload the Kedro project as a tarball and return its S3 URI. The S3 key contains a content hash."""
    files = sorted(
        path
        for entry in PROJECT_FILES
        for path in ([ROOT / entry] if (ROOT / entry).is_file() else (ROOT / entry).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            digest.update(relative.encode() + path.read_bytes())
            tar.add(path, arcname=relative)
    key = f"code/{digest.hexdigest()[:12]}/project.tar.gz"
    session.client("s3").put_object(Bucket=names.bucket, Key=key, Body=buffer.getvalue())
    return f"s3://{names.bucket}/{key}"


def stage_training_source(target: Path) -> None:
    """Copy the Kedro project and train_entry.py into the source dir of the Training job.

    The training toolkit installs requirements.txt from this directory with pip.
    """
    for entry in PROJECT_FILES:
        source = ROOT / entry
        if source.is_dir():
            shutil.copytree(source, target / entry, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(source, target / entry)
    shutil.copy(ROOT / "deploy/train_entry.py", target / "train_entry.py")


def kedro_step(name, kedro_pipeline, *, ctx, inputs, outputs, params=None, property_files=None):
    """Make a Processing step that runs one Kedro pipeline."""
    processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        role=ctx["role"],
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        base_job_name=f"{PREFIX}-{name.lower()}",
        sagemaker_session=ctx["session"],
        env={"KEDRO_DISABLE_TELEMETRY": "1"},
    )
    arguments = ["--pipeline", kedro_pipeline]
    if params is not None:
        arguments += ["--params", params]
    step_args = processor.run(
        code=str(ROOT / "deploy/entrypoint.py"),
        inputs=[
            ProcessingInput(input_name="project", source=ctx["code_uri"], destination="/opt/ml/processing/input/project"),
            *inputs,
        ],
        outputs=outputs,
        arguments=arguments,
    )
    return ProcessingStep(name=name, step_args=step_args, property_files=property_files)


def run_output(names, pipeline_name: str, output: str) -> Join:
    """s3://<bucket>/pipelines/<pipeline>/<execution id>/<output>"""
    return Join(on="/", values=[f"s3://{names.bucket}/pipelines", pipeline_name, ExecutionVariables.PIPELINE_EXECUTION_ID, output])


def build_train_pipeline(names, ctx) -> Pipeline:
    input_data = ParameterString("InputDataUri", default_value=f"s3://{names.bucket}/data/raw")
    model_params = ParameterString("ModelParams", default_value="model.n_estimators=100,model.max_depth=3")
    accuracy_threshold = ParameterFloat("AccuracyThreshold", default_value=0.9)

    preprocess = kedro_step(
        "Preprocess",
        "data_processing",
        ctx=ctx,
        inputs=[ProcessingInput(input_name="raw", source=input_data, destination=f"{DATA}/01_raw")],
        outputs=[
            ProcessingOutput(
                output_name="model_input",
                source=f"{DATA}/05_model_input",
                destination=run_output(names, names.train_pipeline, "model_input"),
            )
        ],
    )

    model_input = preprocess.properties.ProcessingOutputConfig.Outputs["model_input"].S3Output.S3Uri
    estimator = SKLearn(
        entry_point="train_entry.py",
        source_dir=ctx["training_source_dir"],
        framework_version=SKLEARN_VERSION,
        py_version="py3",
        instance_type=TRAINING_INSTANCE_TYPE,
        instance_count=1,
        role=ctx["role"],
        sagemaker_session=ctx["session"],
        base_job_name=f"{PREFIX}-train",
        output_path=f"s3://{names.bucket}/pipelines/{names.train_pipeline}/models",
        code_location=f"s3://{names.bucket}/code",
        hyperparameters={"model-params": model_params},
        environment={"KEDRO_DISABLE_TELEMETRY": "1"},
    )
    train = TrainingStep(name="Train", step_args=estimator.fit(inputs={"model_input": TrainingInput(s3_data=model_input)}))
    model_artifact = train.properties.ModelArtifacts.S3ModelArtifacts

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    evaluate = kedro_step(
        "Evaluate",
        "evaluation",
        ctx=ctx,
        params=model_params,
        inputs=[
            ProcessingInput(input_name="model", source=model_artifact, destination="/opt/ml/processing/model_artifact"),
            ProcessingInput(input_name="model_input", source=model_input, destination=f"{DATA}/05_model_input"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source=f"{DATA}/08_reporting",
                destination=run_output(names, names.train_pipeline, "evaluation"),
            ),
        ],
        property_files=[evaluation_report],
    )
    evaluation_uri = evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri

    def metric(path: str) -> JsonGet:
        return JsonGet(step_name=evaluate.name, property_file=evaluation_report, json_path=path)

    # A new version always starts with no approval. Only a promoter can approve it.
    model = Model(
        image_uri=image_uris.retrieve("sklearn", names.region, version=SKLEARN_VERSION),
        model_data=model_artifact,
        role=ctx["role"],
        sagemaker_session=ctx["session"],
    )
    register = ModelStep(
        name="Register",
        step_args=model.register(
            model_package_group_name=names.model_group,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            approval_status="PendingManualApproval",
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=Join(on="/", values=[evaluation_uri, "evaluation.json"]),
                    content_type="application/json",
                )
            ),
            customer_metadata_properties={
                "accuracy": metric("summary.accuracy"),
                "weighted_f1": metric("summary.weighted_f1"),
                "n_estimators": metric("summary.n_estimators"),
                "max_depth": metric("summary.max_depth"),
                "pipeline_execution_id": ExecutionVariables.PIPELINE_EXECUTION_ID,
            },
        ),
    )

    tag_challenger = LambdaStep(
        name="TagChallenger",
        lambda_func=Lambda(function_arn=names.registry_lambda_arn, session=ctx["session"]),
        inputs={
            "action": "tag_challenger",
            "group": names.model_group,
            "model_package_arn": register.properties.ModelPackageArn,
        },
        outputs=[LambdaOutput("challenger_version", LambdaOutputTypeEnum.String)],
    )

    quality_gate = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=metric("multiclass_classification_metrics.accuracy.value"), right=accuracy_threshold
            )
        ],
        if_steps=[register, tag_challenger],
        else_steps=[
            FailStep(
                name="BelowAccuracyThreshold",
                error_message=Join(
                    on=" ", values=["Accuracy", metric("summary.accuracy"), "is below threshold", accuracy_threshold]
                ),
            )
        ],
    )

    return Pipeline(
        name=names.train_pipeline,
        parameters=[input_data, model_params, accuracy_threshold],
        steps=[preprocess, train, evaluate, quality_gate],
        sagemaker_session=ctx["session"],
    )


def build_inference_pipeline(names, ctx) -> Pipeline:
    input_data = ParameterString("InputDataUri", default_value=f"s3://{names.bucket}/data/inference")
    model_alias = ParameterString("ModelAlias", default_value=CHAMPION, enum_values=["champion", "challenger"])

    # The pipeline finds the model when it runs. Thus, a promotion changes the
    # model of the next execution. You do not deploy this pipeline again.
    resolve = LambdaStep(
        name="ResolveModel",
        lambda_func=Lambda(function_arn=names.registry_lambda_arn, session=ctx["session"]),
        inputs={"action": "resolve", "group": names.model_group, "alias": model_alias},
        outputs=[
            LambdaOutput("model_version", LambdaOutputTypeEnum.String),
            LambdaOutput("model_package_arn", LambdaOutputTypeEnum.String),
            LambdaOutput("model_data_url", LambdaOutputTypeEnum.String),
        ],
    )

    score = kedro_step(
        "Score",
        "inference",
        ctx=ctx,
        params=Join(on="", values=["model_version=", resolve.properties.Outputs["model_version"]]),
        inputs=[
            ProcessingInput(
                input_name="model",
                source=resolve.properties.Outputs["model_data_url"],
                destination="/opt/ml/processing/model_artifact",
            ),
            ProcessingInput(input_name="batch", source=input_data, destination=f"{DATA}/01_raw"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source=f"{DATA}/07_model_output",
                destination=run_output(names, names.inference_pipeline, "predictions"),
            )
        ],
    )

    return Pipeline(
        name=names.inference_pipeline,
        parameters=[input_data, model_alias],
        steps=[resolve, score],
        sagemaker_session=ctx["session"],
    )


def deploy(session, names) -> None:
    role = role_arn(names, names.execution_role)
    with tempfile.TemporaryDirectory() as training_source_dir:
        stage_training_source(Path(training_source_dir))
        ctx = {
            "session": PipelineSession(boto_session=session, default_bucket=names.bucket),
            "role": role,
            "code_uri": package_project(session, names),
            "training_source_dir": training_source_dir,
        }
        print(f"code     {ctx['code_uri']}")
        for build in (build_train_pipeline, build_inference_pipeline):
            pipeline = build(names, ctx)
            pipeline.upsert(role_arn=role)  # this call uploads the training source dir
            print(f"pipeline {pipeline.name}")


def run(session, names, which: str, parameters: dict, wait: bool) -> None:
    sagemaker = session.client("sagemaker")
    pipeline_name = {"train": names.train_pipeline, "inference": names.inference_pipeline}[which]
    arn = sagemaker.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=[{"Name": key, "Value": value} for key, value in parameters.items()],
    )["PipelineExecutionArn"]
    print(f"started  {arn}", flush=True)
    if not wait:
        return

    seen: dict = {}
    while True:
        status = sagemaker.describe_pipeline_execution(PipelineExecutionArn=arn)["PipelineExecutionStatus"]
        steps = sagemaker.list_pipeline_execution_steps(PipelineExecutionArn=arn, SortOrder="Ascending")["PipelineExecutionSteps"]
        for step in steps:
            if seen.get(step["StepName"]) == step["StepStatus"]:
                continue
            seen[step["StepName"]] = step["StepStatus"]
            detail = step.get("FailureReason", "")
            for output in step.get("Metadata", {}).get("Lambda", {}).get("OutputParameters", []):
                detail += f" {output['Name']}={output['Value']}"
            if "RegisterModel" in step.get("Metadata", {}):
                detail += f" {step['Metadata']['RegisterModel']['Arn']}"
            print(f"  {step['StepName']:<28} {step['StepStatus']:<10} {detail.strip()}", flush=True)
        if status not in ("Executing", "Stopping"):
            break
        time.sleep(15)
    print(f"finished {status}")
    if which == "inference" and status == "Succeeded":
        execution_id = arn.rsplit("/", 1)[1]
        print(f"output   s3://{names.bucket}/pipelines/{pipeline_name}/{execution_id}/predictions/predictions.csv")
    if status != "Succeeded":
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deploy", help="package the Kedro project and create/update both pipelines")
    run_parser = commands.add_parser("run", help="start a pipeline execution")
    run_parser.add_argument("pipeline", choices=["train", "inference"])
    run_parser.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE")
    run_parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    session = get_session()
    names = get_names(session)
    if args.command == "deploy":
        deploy(session, names)
    else:
        run(session, names, args.pipeline, dict(item.split("=", 1) for item in args.param), not args.no_wait)


if __name__ == "__main__":
    main()
