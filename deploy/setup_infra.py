"""Make all the resources that the pipelines use: the S3 bucket and data, the IAM
roles, the Model Registry group and the Studio domain. You can run the script
again. It does not make a second copy of a resource.

    python deploy/setup_infra.py
"""

from __future__ import annotations

import json
import time

from botocore.exceptions import ClientError

from common import ROOT, get_names, get_session, role_arn


def trust(principal: dict) -> dict:
    return {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": principal, "Action": "sts:AssumeRole"}]}


def bucket_policy(names) -> dict:
    """Read and write access to the project bucket. The Kedro catalog and the code upload use it."""
    bucket_arn = f"arn:aws:s3:::{names.bucket}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ProjectBucket",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [bucket_arn, f"{bucket_arn}/*"],
            }
        ],
    }


def ensure_bucket(session, names) -> None:
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=names.bucket)
    except ClientError:
        s3.create_bucket(Bucket=names.bucket, CreateBucketConfiguration={"LocationConstraint": names.region})
    s3.put_public_access_block(
        Bucket=names.bucket,
        PublicAccessBlockConfiguration={"BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True},
    )
    # The same layout as the local data folder. conf/sagemaker/globals.yml sets data_root to s3://<bucket>/data.
    s3.upload_file(str(ROOT / "data/01_raw/iris.csv"), names.bucket, "data/01_raw/iris.csv")
    s3.upload_file(str(ROOT / "data/01_raw/iris_batch.csv"), names.bucket, "data/01_raw/iris_batch.csv")
    print(f"bucket   s3://{names.bucket} (data uploaded to data/01_raw)")


def ensure_role(iam, name: str, trust_policy: dict, description: str, inline: dict, managed: tuple = ()) -> None:
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust_policy), Description=description)
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust_policy))
    for arn in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=arn)
    iam.put_role_policy(RoleName=name, PolicyName=f"{name}-policy", PolicyDocument=json.dumps(inline))
    print(f"role     {name}")


def ensure_model_group(session, names) -> None:
    sagemaker = session.client("sagemaker")
    try:
        sagemaker.create_model_package_group(
            ModelPackageGroupName=names.model_group,
            ModelPackageGroupDescription="Iris classifier. The 'champion' tag of the group contains the version that evaluate and inference use.",
        )
    except ClientError as error:
        if "already exists" not in str(error):
            raise
    print(f"registry {names.model_group}")


def ensure_studio(session, names) -> None:
    """Make the SageMaker Studio domain. Studio shows the pipeline graphs and the Model Registry.

    The domain uses public internet access. Thus, it does not use a NAT gateway
    and has no fixed cost. AWS bills Studio apps and spaces only while they run.
    """
    sagemaker, ec2 = session.client("sagemaker"), session.client("ec2")
    domains = {d["DomainName"]: d["DomainId"] for d in sagemaker.list_domains()["Domains"]}
    domain_id = domains.get(names.studio_domain)
    if not domain_id:
        vpc_id = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
        subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
        domain_arn = sagemaker.create_domain(
            DomainName=names.studio_domain,
            AuthMode="IAM",
            DefaultUserSettings={"ExecutionRole": role_arn(names, names.studio_role)},
            VpcId=vpc_id,
            SubnetIds=[subnet["SubnetId"] for subnet in subnets],
            AppNetworkAccessType="PublicInternetOnly",
        )["DomainArn"]
        domain_id = domain_arn.rsplit("/", 1)[1]
    while (status := sagemaker.describe_domain(DomainId=domain_id)["Status"]) not in ("InService", "Failed"):
        time.sleep(15)
    if status == "Failed":
        raise RuntimeError(sagemaker.describe_domain(DomainId=domain_id).get("FailureReason", "Studio domain failed"))
    try:
        sagemaker.create_user_profile(DomainId=domain_id, UserProfileName=names.studio_user)
    except ClientError as error:
        if error.response["Error"]["Code"] != "ResourceInUse":
            raise
    print(f"studio   {names.studio_domain} ({domain_id}), user profile {names.studio_user}")


def main() -> None:
    session = get_session()
    names = get_names(session)
    iam = session.client("iam")
    sagemaker_full_access = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"

    ensure_bucket(session, names)
    ensure_role(
        iam,
        names.execution_role,
        trust({"Service": "sagemaker.amazonaws.com"}),
        "Runs the iris demo SageMaker pipelines. The Kedro jobs register models and set the champion tag.",
        bucket_policy(names),
        managed=(sagemaker_full_access,),
    )
    ensure_model_group(session, names)
    ensure_role(
        iam,
        names.studio_role,
        trust({"Service": "sagemaker.amazonaws.com"}),
        "SageMaker Studio users: view and run pipelines, browse the registry.",
        bucket_policy(names),
        managed=(sagemaker_full_access,),
    )
    time.sleep(10)  # wait until AWS makes the new role available to the domain
    ensure_studio(session, names)


if __name__ == "__main__":
    main()
