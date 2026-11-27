"""Delete every AWS resource created by setup_infra.py and pipelines.py.

    python deploy/teardown.py --yes
"""

from __future__ import annotations

import argparse
import time

from botocore.exceptions import ClientError

from common import PREFIX, get_names, get_session

# Resources of earlier versions of this project. The script deletes them too, if they exist.
LEGACY_ROLES = (f"{PREFIX}-model-promoter", f"{PREFIX}-data-scientist")
LEGACY_GROUP = f"{PREFIX}-model-promoters"


def ignore_missing(call, **kwargs) -> None:
    try:
        call(**kwargs)
    except ClientError as error:
        code = error.response["Error"]["Code"]
        if code not in ("ResourceNotFound", "ResourceNotFoundException", "NoSuchEntity", "NoSuchBucket", "ValidationException", "RepositoryNotFoundException"):
            raise


def delete_role(iam, role: str) -> None:
    try:
        for attached in iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=role, PolicyArn=attached["PolicyArn"])
        for inline in iam.list_role_policies(RoleName=role)["PolicyNames"]:
            iam.delete_role_policy(RoleName=role, PolicyName=inline)
        iam.delete_role(RoleName=role)
    except iam.exceptions.NoSuchEntityException:
        pass


def delete_group(iam, group: str) -> None:
    try:
        for user in iam.get_group(GroupName=group)["Users"]:
            iam.remove_user_from_group(GroupName=group, UserName=user["UserName"])
        for policy in iam.list_group_policies(GroupName=group)["PolicyNames"]:
            iam.delete_group_policy(GroupName=group, PolicyName=policy)
        iam.delete_group(GroupName=group)
    except iam.exceptions.NoSuchEntityException:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", required=True, help="confirm deletion")
    parser.parse_args()

    session = get_session()
    names = get_names(session)
    sagemaker, iam, ecr = session.client("sagemaker"), session.client("iam"), session.client("ecr")

    for pipeline in names.pipelines.values():
        ignore_missing(sagemaker.delete_pipeline, PipelineName=pipeline)
    ignore_missing(ecr.delete_repository, repositoryName=names.ecr_repository, force=True)

    try:
        packages = sagemaker.list_model_packages(ModelPackageGroupName=names.model_group)["ModelPackageSummaryList"]
    except ClientError:
        packages = []
    for package in packages:
        sagemaker.delete_model_package(ModelPackageName=package["ModelPackageArn"])
    ignore_missing(sagemaker.delete_model_package_group, ModelPackageGroupName=names.model_group)

    for domain in sagemaker.list_domains()["Domains"]:
        if domain["DomainName"] != names.studio_domain:
            continue
        domain_id = domain["DomainId"]
        # This code does not delete Studio apps or spaces. Delete them in Studio first.
        ignore_missing(sagemaker.delete_user_profile, DomainId=domain_id, UserProfileName=names.studio_user)
        while sagemaker.list_user_profiles(DomainIdEquals=domain_id)["UserProfiles"]:
            time.sleep(10)
        sagemaker.delete_domain(DomainId=domain_id, RetentionPolicy={"HomeEfsFileSystem": "Delete"})

    for role in (names.execution_role, names.studio_role, *LEGACY_ROLES):
        delete_role(iam, role)
    delete_group(iam, LEGACY_GROUP)

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
