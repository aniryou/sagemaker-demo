"""Runs inside a SageMaker Processing job: unpack the Kedro project, install its
requirements and execute one Kedro pipeline with the `sagemaker` environment.

Uses only the standard library, because it starts before anything is installed.
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

BASE = Path("/opt/ml/processing")
PROJECT_TAR = BASE / "input/project/project.tar.gz"
PROJECT_DIR = Path("/tmp/project")
MODEL_ARTIFACT = BASE / "model_artifact"  # model.tar.gz from the training job / registry
MODEL_DIR = BASE / "data/06_models"  # where the Kedro catalog reads/writes model.joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--params", default="")
    args = parser.parse_args()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PROJECT_TAR) as tar:
        tar.extractall(PROJECT_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", "requirements.txt"],
        cwd=PROJECT_DIR,
        check=True,
    )

    registered_model = MODEL_ARTIFACT / "model.tar.gz"
    if registered_model.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(registered_model) as tar:
            tar.extractall(MODEL_DIR)

    command = [sys.executable, "-m", "kedro", "run", "--env", "sagemaker", "--pipeline", args.pipeline]
    if args.params:
        command += ["--params", args.params]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    main()
