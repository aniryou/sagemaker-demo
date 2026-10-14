"""Lambda used by the SageMaker pipelines to read and write registry aliases.

Aliases are tags on the model package group whose value is a model version:
  champion   -> version the inference pipeline uses (only promoters may set it)
  challenger -> newest version that passed the quality gate (set by training)

This function's role can only write the `challenger` tag.
"""

import boto3

sagemaker = boto3.client("sagemaker")


def _group_arn(group):
    return sagemaker.describe_model_package_group(ModelPackageGroupName=group)["ModelPackageGroupArn"]


def _aliases(group_arn):
    return {tag["Key"]: tag["Value"] for tag in sagemaker.list_tags(ResourceArn=group_arn)["Tags"]}


def resolve(event):
    """Return the model version an alias points at."""
    group, alias = event["group"], event.get("alias", "champion")
    group_arn = _group_arn(group)
    version = _aliases(group_arn).get(alias)
    if not version:
        raise RuntimeError(
            f"No '{alias}' model in registry group '{group}'. "
            "An authorized promoter must run: python deploy/promote.py promote"
        )
    package_arn = group_arn.replace(":model-package-group/", ":model-package/") + f"/{version}"
    package = sagemaker.describe_model_package(ModelPackageName=package_arn)
    status = package["ModelApprovalStatus"]
    if alias == "champion" and status != "Approved":
        raise RuntimeError(f"Champion version {version} has approval status '{status}', expected 'Approved'.")
    return {
        "model_version": str(version),
        "model_package_arn": package_arn,
        "model_data_url": package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
    }


def tag_challenger(event):
    """Point the `challenger` alias at a freshly registered version."""
    version = event["model_package_arn"].rsplit("/", 1)[1]
    sagemaker.add_tags(ResourceArn=_group_arn(event["group"]), Tags=[{"Key": "challenger", "Value": version}])
    return {"challenger_version": version}


def handler(event, context):
    return {"resolve": resolve, "tag_challenger": tag_challenger}[event["action"]](event)
