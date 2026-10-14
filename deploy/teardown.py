"""Delete every AWS resource created by setup_infra.py and pipelines.py.

    python deploy/teardown.py --yes
"""

from __future__ import annotations

import argparse
import time

from botocore.exceptions import ClientError

from common import get_names, get_session


def ignore_missing(call, **kwargs) -> None:
    try:
        call(**kwargs)
    except ClientError as error:
        code = error.response["Error"]["Code"]
        if code not in ("ResourceNotFound", "ResourceNotFoundException", "NoSuchEntity", "NoSuchBucket", "ValidationException"):
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", required=True, help="confirm deletion")
    parser.parse_args()

    session = get_session()
    names = get_names(session)
    sagemaker, iam = session.client("sagemaker"), session.client("iam")

    for pipeline in (names.train_pipeline, names.inference_pipeline):
        ignore_missing(sagemaker.delete_pipeline, PipelineName=pipeline)

    try:
        packages = sagemaker.list_model_packages(ModelPackageGroupName=names.model_group)["ModelPackageSummaryList"]
    except ClientError:
        packages = []
    for package in packages:
        sagemaker.delete_model_package(ModelPackageName=package["ModelPackageArn"])
    ignore_missing(sagemaker.delete_model_package_group, ModelPackageGroupName=names.model_group)

    ignore_missing(session.client("lambda").delete_function, FunctionName=names.registry_lambda)

    for domain in sagemaker.list_domains()["Domains"]:
        if domain["DomainName"] != names.studio_domain:
            continue
        domain_id = domain["DomainId"]
        # This code does not delete Studio apps or spaces. Delete them in Studio first.
        ignore_missing(sagemaker.delete_user_profile, DomainId=domain_id, UserProfileName=names.studio_user)
        while sagemaker.list_user_profiles(DomainIdEquals=domain_id)["UserProfiles"]:
            time.sleep(10)
        sagemaker.delete_domain(DomainId=domain_id, RetentionPolicy={"HomeEfsFileSystem": "Delete"})

    for role in (names.execution_role, names.lambda_role, names.promoter_role, names.data_scientist_role, names.studio_role):
        try:
            for attached in iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=role, PolicyArn=attached["PolicyArn"])
            for inline in iam.list_role_policies(RoleName=role)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role, PolicyName=inline)
            iam.delete_role(RoleName=role)
        except iam.exceptions.NoSuchEntityException:
            pass

    try:
        for user in iam.get_group(GroupName=names.promoters_group)["Users"]:
            iam.remove_user_from_group(GroupName=names.promoters_group, UserName=user["UserName"])
        iam.delete_group_policy(GroupName=names.promoters_group, PolicyName="assume-model-promoter")
        iam.delete_group(GroupName=names.promoters_group)
    except iam.exceptions.NoSuchEntityException:
        pass

    bucket = session.resource("s3").Bucket(names.bucket)
    try:
        bucket.objects.all().delete()
        bucket.delete()
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchBucket":
            raise
    print("deleted all iris-demo resources")


if __name__ == "__main__":
    main()
