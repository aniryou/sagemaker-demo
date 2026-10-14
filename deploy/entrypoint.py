"""Entry point of a SageMaker Processing job.

The script unpacks the Kedro project, installs its requirements and runs one
Kedro pipeline with the `sagemaker` environment. For inference, the script
also finds the model of an alias in the Model Registry and downloads it.

At the start, the script uses only the standard library, because no other
package is installed then.
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

BASE = Path("/opt/ml/processing")
PROJECT_TAR = BASE / "input/project/project.tar.gz"
PROJECT_DIR = Path("/tmp/project")
MODEL_ARTIFACT = BASE / "model_artifact"  # model.tar.gz from the training job or the registry
MODEL_DIR = BASE / "data/06_models"  # the Kedro catalog reads and writes model.joblib here


def resolve_model(group: str, alias: str) -> tuple:
    """Return the version and the model.tar.gz URI of an alias.

    champion   -> the version in the `champion` tag of the group. Its status must be Approved.
    challenger -> the newest version with the status PendingManualApproval.
    """
    import boto3  # the container image contains boto3

    sagemaker = boto3.client("sagemaker")
    group_arn = sagemaker.describe_model_package_group(ModelPackageGroupName=group)["ModelPackageGroupArn"]
    if alias == "champion":
        tags = {tag["Key"]: tag["Value"] for tag in sagemaker.list_tags(ResourceArn=group_arn)["Tags"]}
        if "champion" not in tags:
            sys.exit(f"No champion in registry group '{group}'. A promoter must run: python deploy/promote.py promote")
        package_arn = group_arn.replace(":model-package-group/", ":model-package/") + "/" + tags["champion"]
    else:
        pending = sagemaker.list_model_packages(
            ModelPackageGroupName=group,
            ModelApprovalStatus="PendingManualApproval",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )["ModelPackageSummaryList"]
        if not pending:
            sys.exit(f"No challenger in registry group '{group}'. Run the train pipeline first.")
        package_arn = pending[0]["ModelPackageArn"]

    package = sagemaker.describe_model_package(ModelPackageName=package_arn)
    version = str(package["ModelPackageVersion"])
    if alias == "champion" and package["ModelApprovalStatus"] != "Approved":
        sys.exit(f"Champion version {version} has the status {package['ModelApprovalStatus']}, not Approved.")
    return version, package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]


def download(s3_uri: str, target: Path) -> None:
    import boto3

    bucket, key = s3_uri.replace("s3://", "", 1).split("/", 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--params", default="")
    parser.add_argument("--resolve-alias", choices=["champion", "challenger"], help="download the model of this alias")
    parser.add_argument("--model-group", help="model package group of --resolve-alias")
    args = parser.parse_args()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PROJECT_TAR) as tar:
        tar.extractall(PROJECT_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", "requirements.txt"],
        cwd=PROJECT_DIR,
        check=True,
    )

    params = args.params
    registered_model = MODEL_ARTIFACT / "model.tar.gz"
    if args.resolve_alias:
        version, model_uri = resolve_model(args.model_group, args.resolve_alias)
        print(f"Resolved {args.resolve_alias} -> version {version} ({model_uri})", flush=True)
        download(model_uri, registered_model)
        params = ",".join(filter(None, [f"model_version={version}", params]))

    if registered_model.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(registered_model) as tar:
            tar.extractall(MODEL_DIR)

    command = [sys.executable, "-m", "kedro", "run", "--env", "sagemaker", "--pipeline", args.pipeline]
    if params:
        command += ["--params", params]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    main()
