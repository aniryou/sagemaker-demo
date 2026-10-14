"""Entry point of a SageMaker Processing job.

The script unpacks the Kedro project, installs its requirements and runs one
Kedro pipeline with the `sagemaker` environment. The script uses only the
standard library, because no other package is installed when it starts.
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
