"""Define, deploy and run the SageMaker pipelines.

    python deploy/pipelines.py deploy
    python deploy/pipelines.py run train
    python deploy/pipelines.py run train -p KedroParams=model.n_estimators=200,model.max_depth=4
    python deploy/pipelines.py run evaluate
    python deploy/pipelines.py run inference

Each SageMaker pipeline has one Processing step. The step runs the project image
(see the Dockerfile) with the Kedro pipeline of the same name:
`kedro run --env sagemaker --pipeline <name>`. The Kedro catalog reads and writes
S3 and the Model Registry. Thus, the SageMaker pipeline has no inputs and no
outputs. `deploy` pins the digest of the `latest` image in the pipelines.
"""

from __future__ import annotations

import argparse
import sys
import time

from botocore.exceptions import ClientError
from sagemaker.processing import Processor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

from common import CHAMPION, PIPELINES, PREFIX, get_names, get_session, role_arn

INSTANCE_TYPE = "ml.t3.medium"  # The account has a Processing quota only for ml.t3.


def resolve_image(session, names) -> str:
    """Return the image URI with the digest of the `latest` tag, for example <repository>@sha256:..."""
    try:
        details = session.client("ecr").describe_images(
            repositoryName=names.ecr_repository, imageIds=[{"imageTag": "latest"}]
        )["imageDetails"][0]
    except ClientError as error:
        if error.response["Error"]["Code"] in ("RepositoryNotFoundException", "ImageNotFoundException"):
            sys.exit(f"No image '{names.image_repository}:latest' in ECR. Run: python deploy/build_image.py")
        raise
    return f"{names.image_repository}@{details['imageDigest']}"


def build_pipeline(name: str, names, ctx) -> Pipeline:
    """Make the SageMaker pipeline that runs the Kedro pipeline `name`."""
    kedro_params = ParameterString("KedroParams", default_value="")  # for example: model.n_estimators=200
    processor = Processor(
        image_uri=ctx["image_uri"],  # its entry point is `kedro run --env sagemaker`
        role=ctx["role"],
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        base_job_name=f"{PREFIX}-{name}",
        sagemaker_session=ctx["session"],
        env={
            "IRIS_BUCKET": names.bucket,  # conf/sagemaker/globals.yml reads it
            "AWS_DEFAULT_REGION": names.region,
        },
    )
    step_args = processor.run(arguments=["--pipeline", name, "--params", kedro_params])
    return Pipeline(
        name=names.pipelines[name],
        parameters=[kedro_params],
        steps=[ProcessingStep(name=f"kedro-run-{name}", step_args=step_args)],
        sagemaker_session=ctx["session"],
    )


def deploy(session, names) -> None:
    role = role_arn(names, names.execution_role)
    ctx = {
        "session": PipelineSession(boto_session=session, default_bucket=names.bucket),
        "role": role,
        "image_uri": resolve_image(session, names),
    }
    print(f"image    {ctx['image_uri']}")
    for name in PIPELINES:
        pipeline = build_pipeline(name, names, ctx)
        pipeline.upsert(role_arn=role)
        print(f"pipeline {pipeline.name}")


def champion(session, names) -> tuple:
    """Return the champion version and its registry entry, or (None, {})."""
    sagemaker = session.client("sagemaker")
    tags = {tag["Key"]: tag["Value"] for tag in sagemaker.list_tags(ResourceArn=names.model_group_arn)["Tags"]}
    version = tags.get(CHAMPION)
    if not version:
        return None, {}
    return version, sagemaker.describe_model_package(ModelPackageName=f"{names.model_package_arn_prefix}/{version}")


def print_result(session, names, which: str) -> None:
    version, package = champion(session, names)
    if which == "inference":
        print(f"output   s3://{names.bucket}/data/07_model_output/predictions.csv (champion version {version})")
        return
    if not version:
        print("champion (none)")
        return
    print(f"champion version {version}  {package['ModelApprovalStatus']}  {package.get('SourceUri', '')}")
    for key, value in sorted(package.get("CustomerMetadataProperties", {}).items()):
        print(f"  {key:<14} {value}")


def run(session, names, which: str, parameters: dict, wait: bool) -> None:
    sagemaker = session.client("sagemaker")
    arn = sagemaker.start_pipeline_execution(
        PipelineName=names.pipelines[which],
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
            print(f"  {step['StepName']:<24} {step['StepStatus']:<10} {step.get('FailureReason', '')}".rstrip(), flush=True)
        if status not in ("Executing", "Stopping"):
            break
        time.sleep(15)
    print(f"finished {status}")
    if status != "Succeeded":
        sys.exit(1)
    print_result(session, names, which)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deploy", help="create or update the three pipelines with the latest image")
    run_parser = commands.add_parser("run", help="start a pipeline execution and wait for it")
    run_parser.add_argument("pipeline", choices=PIPELINES)
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
