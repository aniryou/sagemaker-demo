# Image of the Kedro project. Each SageMaker pipeline runs one Processing job
# with this image: `kedro run --env sagemaker --pipeline <name>`.
#
#   python deploy/build_image.py     # build the image and push it to ECR
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KEDRO_DISABLE_TELEMETRY=1

WORKDIR /opt/iris_demo

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY conf/base conf/base
COPY conf/sagemaker conf/sagemaker
COPY src src
RUN mkdir -p conf/local

# SageMaker Pipelines passes the arguments: --pipeline <name> --params <KedroParams>
ENTRYPOINT ["kedro", "run", "--env", "sagemaker"]
