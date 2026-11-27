"""Entry point of a SageMaker Processing job.

The script unpacks the Kedro project, installs its requirements and runs one
Kedro pipeline with the `sagemaker` environment. The Kedro catalog reads and
writes S3 and the Model Registry. Thus, the job has no data inputs or outputs.

At the start, the script uses only the standard library, because no other
package is installed then.
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_TAR = Path("/opt/ml/processing/input/project/project.tar.gz")
PROJECT_DIR = Path("/tmp/project")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="name of the Kedro pipeline")
    parser.add_argument("--params", default="", help="value of `kedro run --params`")
    args = parser.parse_args()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(PROJECT_TAR) as tar:
        tar.extractall(PROJECT_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", "requirements.txt"],
        cwd=PROJECT_DIR,
        check=True,
    )

    command = [sys.executable, "-m", "kedro", "run", "--env", "sagemaker", "--pipeline", args.pipeline]
    if args.params.strip():
        command += ["--params", args.params.strip()]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    main()
