"""Build the project image and push it to ECR. The SageMaker pipelines run this image.

    python deploy/build_image.py

The script tags the image with a hash of its content and with `latest`. Then
`python deploy/pipelines.py deploy` pins the digest of `latest` in the pipelines.

The script needs Docker. On a computer without Docker, the package
sagemaker-studio-image-build builds the same Dockerfile in AWS CodeBuild:

    pip install sagemaker-studio-image-build
    sm-docker build . --repository iris-demo:latest
"""

from __future__ import annotations

import base64
import hashlib
import subprocess

from common import ROOT, get_names, get_session

IMAGE_FILES = ("Dockerfile", "requirements.txt", "pyproject.toml", "conf/base", "conf/sagemaker", "src/iris_demo")


def content_tag() -> str:
    """A tag that changes when a file of the image changes."""
    digest = hashlib.sha256()
    for entry in IMAGE_FILES:
        base = ROOT / entry
        for path in sorted([base] if base.is_file() else base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.relative_to(ROOT).as_posix().encode() + path.read_bytes())
    return digest.hexdigest()[:12]


def docker(*args: str, **kwargs) -> None:
    subprocess.run(["docker", *args], check=True, **kwargs)


def ensure_repository(ecr, name: str) -> None:
    try:
        ecr.create_repository(repositoryName=name, imageScanningConfiguration={"scanOnPush": True})
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass


def main() -> None:
    session = get_session()
    names = get_names(session)
    ecr = session.client("ecr")
    ensure_repository(ecr, names.ecr_repository)

    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    registry = auth["proxyEndpoint"].removeprefix("https://")
    docker("login", "--username", user, "--password-stdin", registry, input=password.encode())

    tag = content_tag()
    image, latest = f"{names.image_repository}:{tag}", f"{names.image_repository}:latest"
    # SageMaker Processing instances are x86_64. The platform flag matters on an ARM computer.
    docker("build", "--platform", "linux/amd64", "-t", image, "-t", latest, str(ROOT))
    docker("push", image)
    docker("push", latest)
    digest = ecr.describe_images(repositoryName=names.ecr_repository, imageIds=[{"imageTag": tag}])["imageDetails"][0]["imageDigest"]
    print(f"image    {image}")
    print(f"digest   {digest}")


if __name__ == "__main__":
    main()
