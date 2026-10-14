"""Create (idempotently) everything the pipelines need: S3 bucket + data, IAM
roles, the registry-helper Lambda and the Model Registry group.

    python deploy/setup_infra.py
"""

from __future__ import annotations

import io
import json
import time
import zipfile

from botocore.exceptions import ClientError

from common import CHALLENGER, CHAMPION, PREVIOUS_CHAMPION, ROOT, get_names, get_session, role_arn


def trust(principal: dict) -> dict:
    return {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": principal, "Action": "sts:AssumeRole"}]}


def policy(*statements: dict) -> dict:
    return {"Version": "2012-10-17", "Statement": list(statements)}


def build_policies(names) -> dict:
    """IAM policies. The champion/challenger authorization model lives here."""
    packages = f"{names.model_package_arn_prefix}/*"
    bucket_arn = f"arn:aws:s3:::{names.bucket}"
    pipelines = [
        f"arn:aws:sagemaker:{names.region}:{names.account}:pipeline/{name}*"
        for name in (names.train_pipeline, names.inference_pipeline)
    ]

    read_only = {
        "Sid": "ReadRegistryAndPipelines",
        "Effect": "Allow",
        "Action": ["sagemaker:List*", "sagemaker:Describe*", "sagemaker:Search"],
        "Resource": "*",
    }
    # Explicit denies win over any allow, so these identities can never move an
    # alias or approve a version even if a broad managed policy is attached.
    # Scoped to the alias keys: SageMaker Pipelines adds its own bookkeeping tags
    # to the group when it registers a model, and that must keep working.
    cannot_promote = [
        {
            "Sid": "OnlyPromotersMoveAliases",
            "Effect": "Deny",
            "Action": ["sagemaker:AddTags", "sagemaker:DeleteTags"],
            "Resource": names.model_group_arn,
            "Condition": {"ForAnyValue:StringEquals": {"aws:TagKeys": [CHAMPION, PREVIOUS_CHAMPION, CHALLENGER]}},
        },
        {"Sid": "OnlyPromotersApprove", "Effect": "Deny", "Action": "sagemaker:UpdateModelPackage", "Resource": packages},
    ]

    return {
        "promoter": policy(
            read_only,
            {
                "Sid": "MoveModelAliases",
                "Effect": "Allow",
                "Action": "sagemaker:AddTags",
                "Resource": names.model_group_arn,
                "Condition": {"ForAllValues:StringEquals": {"aws:TagKeys": [CHAMPION, PREVIOUS_CHAMPION, CHALLENGER]}},
            },
            {"Sid": "ApproveOrRejectVersions", "Effect": "Allow", "Action": "sagemaker:UpdateModelPackage", "Resource": packages},
        ),
        "data_scientist": policy(
            read_only,
            {
                "Sid": "RunPipelines",
                "Effect": "Allow",
                "Action": ["sagemaker:StartPipelineExecution", "sagemaker:StopPipelineExecution"],
                "Resource": pipelines,
            },
            {"Sid": "ReadArtifacts", "Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"], "Resource": [bucket_arn, f"{bucket_arn}/*"]},
            *cannot_promote,
        ),
        "execution": policy(
            {"Sid": "Artifacts", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], "Resource": [bucket_arn, f"{bucket_arn}/*"]},
            {"Sid": "RegistryHelper", "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": names.registry_lambda_arn},
            *cannot_promote,
        ),
        # Studio is for viewing pipelines/registry and running pipelines. Its
        # users get the same "cannot promote" denies as data scientists.
        "studio": policy(
            {"Sid": "Artifacts", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], "Resource": [bucket_arn, f"{bucket_arn}/*"]},
            *cannot_promote,
        ),
        "lambda": policy(
            {"Sid": "Logs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": f"arn:aws:logs:{names.region}:{names.account}:*"},
            {"Sid": "ReadAliases", "Effect": "Allow", "Action": ["sagemaker:DescribeModelPackageGroup", "sagemaker:ListTags"], "Resource": names.model_group_arn},
            {"Sid": "ReadVersions", "Effect": "Allow", "Action": "sagemaker:DescribeModelPackage", "Resource": packages},
            {
                "Sid": "SetChallengerOnly",
                "Effect": "Allow",
                "Action": "sagemaker:AddTags",
                "Resource": names.model_group_arn,
                "Condition": {"ForAllValues:StringEquals": {"aws:TagKeys": [CHALLENGER]}},
            },
        ),
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
    s3.upload_file(str(ROOT / "data/01_raw/iris.csv"), names.bucket, "data/raw/iris.csv")
    s3.upload_file(str(ROOT / "data/01_raw/iris_batch.csv"), names.bucket, "data/inference/iris_batch.csv")
    print(f"bucket   s3://{names.bucket} (data uploaded)")


def ensure_role(iam, name: str, trust_policy: dict, description: str, inline: dict, managed: tuple = ()) -> None:
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust_policy), Description=description)
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust_policy))
    for arn in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=arn)
    iam.put_role_policy(RoleName=name, PolicyName=f"{name}-policy", PolicyDocument=json.dumps(inline))
    print(f"role     {name}")


def ensure_promoters_group(iam, names) -> None:
    """Members of this IAM group are the 'authorized users' who may promote."""
    try:
        iam.create_group(GroupName=names.promoters_group)
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    iam.put_group_policy(
        GroupName=names.promoters_group,
        PolicyName="assume-model-promoter",
        PolicyDocument=json.dumps(
            policy({"Effect": "Allow", "Action": "sts:AssumeRole", "Resource": role_arn(names, names.promoter_role)})
        ),
    )
    print(f"group    {names.promoters_group}")


def ensure_lambda(session, names) -> None:
    client = session.client("lambda")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "deploy/registry_lambda.py", "registry_lambda.py")
    code = buffer.getvalue()
    try:
        client.get_function(FunctionName=names.registry_lambda)
        client.update_function_code(FunctionName=names.registry_lambda, ZipFile=code)
    except client.exceptions.ResourceNotFoundException:
        for attempt in range(12):  # a new IAM role takes a few seconds to become assumable
            try:
                client.create_function(
                    FunctionName=names.registry_lambda,
                    Runtime="python3.12",
                    Role=role_arn(names, names.lambda_role),
                    Handler="registry_lambda.handler",
                    Code={"ZipFile": code},
                    Timeout=30,
                    Description="Resolves and sets champion/challenger aliases for the iris demo pipelines",
                )
                break
            except client.exceptions.InvalidParameterValueException:
                if attempt == 11:
                    raise
                time.sleep(5)
    client.get_waiter("function_active_v2").wait(FunctionName=names.registry_lambda)
    print(f"lambda   {names.registry_lambda}")


def ensure_model_group(session, names) -> None:
    sagemaker = session.client("sagemaker")
    try:
        sagemaker.create_model_package_group(
            ModelPackageGroupName=names.model_group,
            ModelPackageGroupDescription="Iris classifier. Tags 'champion'/'challenger' hold the version each alias points at.",
        )
    except ClientError as error:
        if "already exists" not in str(error):
            raise
    print(f"registry {names.model_group}")


def ensure_studio(session, names) -> None:
    """SageMaker Studio domain: the UI for pipeline graphs and the Model Registry.

    Public internet access avoids a NAT gateway, so the domain has no fixed
    cost. Only running Studio apps/spaces are billed.
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
    policies = build_policies(names)
    account_principals = {"AWS": f"arn:aws:iam::{names.account}:root"}

    ensure_bucket(session, names)
    ensure_role(
        iam,
        names.execution_role,
        trust({"Service": "sagemaker.amazonaws.com"}),
        "Runs the iris demo SageMaker pipelines. Cannot move champion/challenger aliases.",
        policies["execution"],
        managed=("arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",),
    )
    ensure_role(
        iam,
        names.lambda_role,
        trust({"Service": "lambda.amazonaws.com"}),
        "Registry helper Lambda. May only set the challenger alias.",
        policies["lambda"],
    )
    ensure_role(
        iam,
        names.promoter_role,
        trust(account_principals),
        "Authorized to promote a model version to champion.",
        policies["promoter"],
    )
    ensure_role(
        iam,
        names.data_scientist_role,
        trust(account_principals),
        "Can run pipelines and read the registry, but not promote.",
        policies["data_scientist"],
    )
    ensure_promoters_group(iam, names)
    ensure_model_group(session, names)
    ensure_lambda(session, names)
    ensure_role(
        iam,
        names.studio_role,
        trust({"Service": "sagemaker.amazonaws.com"}),
        "SageMaker Studio users: view and run pipelines, browse the registry. Cannot promote.",
        policies["studio"],
        managed=("arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",),
    )
    time.sleep(10)  # let a newly created role propagate before the domain uses it
    ensure_studio(session, names)


if __name__ == "__main__":
    main()
