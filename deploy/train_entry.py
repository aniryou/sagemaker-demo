"""SageMaker Training job entry point: run the Kedro `training` pipeline.

The training toolkit has already pip-installed requirements.txt from this
source directory (/opt/ml/code) before calling this script.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA = Path("/opt/ml/processing/data")  # data root of the Kedro `sagemaker` env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-params", default="")
    args, _ = parser.parse_known_args()
    params = args.model_params.strip('"')

    # Point the catalog's model_input folder at the training channel.
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "05_model_input").symlink_to(os.environ["SM_CHANNEL_MODEL_INPUT"])

    command = [sys.executable, "-m", "kedro", "run", "--env", "sagemaker", "--pipeline", "training"]
    if params:
        command += ["--params", params]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)

    # SageMaker tars SM_MODEL_DIR into model.tar.gz, the artifact the registry stores.
    shutil.copy(DATA / "06_models/model.joblib", Path(os.environ["SM_MODEL_DIR"]) / "model.joblib")


if __name__ == "__main__":
    main()
