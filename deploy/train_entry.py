"""Entry point of the SageMaker Training job. The script runs the Kedro `training` pipeline.

Before it starts this script, the training toolkit installs requirements.txt
from this source directory (/opt/ml/code) with pip.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA = Path("/opt/ml/processing/data")  # data root of the Kedro `sagemaker` environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-params", default="")
    args, _ = parser.parse_known_args()
    params = args.model_params.strip('"')

    # Link the model_input folder of the catalog to the training channel.
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "05_model_input").symlink_to(os.environ["SM_CHANNEL_MODEL_INPUT"])

    command = [sys.executable, "-m", "kedro", "run", "--env", "sagemaker", "--pipeline", "training"]
    if params:
        command += ["--params", params]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)

    # SageMaker puts SM_MODEL_DIR into model.tar.gz. The registry keeps this file.
    shutil.copy(DATA / "06_models/model.joblib", Path(os.environ["SM_MODEL_DIR"]) / "model.joblib")


if __name__ == "__main__":
    main()
