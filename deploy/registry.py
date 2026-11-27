"""Show the model versions of the registry group, and move the champion tag.

    python deploy/registry.py list
    python deploy/registry.py promote 2        # champion = version 2

The train pipeline sets the champion tag itself. A run with
`KedroParams=model_tag=challenger` registers a challenger instead and leaves
the champion as it is. Use `promote` when that challenger must become the champion.
"""

from __future__ import annotations

import argparse
import sys

from botocore.exceptions import ClientError

from common import CHAMPION, get_names, get_session

CHALLENGER = "challenger"


def tags_of(sagemaker, names) -> dict:
    return {tag["Key"]: tag["Value"] for tag in sagemaker.list_tags(ResourceArn=names.model_group_arn)["Tags"]}


def list_versions(sagemaker, names) -> None:
    tags = tags_of(sagemaker, names)
    summaries = sagemaker.list_model_packages(
        ModelPackageGroupName=names.model_group, SortBy="CreationTime", SortOrder="Descending"
    )["ModelPackageSummaryList"]
    print(f"{'version':<8}{'tags':<22}{'approval':<24}{'accuracy':<10}{'weighted_f1':<13}{'created':<18}source")
    for summary in summaries:
        package = sagemaker.describe_model_package(ModelPackageName=summary["ModelPackageArn"])
        version = str(package["ModelPackageVersion"])
        metadata = package.get("CustomerMetadataProperties", {})
        aliases = ", ".join(key for key, value in sorted(tags.items()) if value == version)
        print(
            f"{version:<8}{aliases:<22}{package['ModelApprovalStatus']:<24}"
            f"{metadata.get('accuracy', '-'):<10}{metadata.get('weighted_f1', '-'):<13}"
            f"{package['CreationTime']:%Y-%m-%d %H:%M}  {package.get('SourceUri', '')}"
        )
    if not summaries:
        print("(no versions registered yet - run the train pipeline)")


def promote(sagemaker, names, version: str) -> None:
    try:
        sagemaker.describe_model_package(ModelPackageName=f"{names.model_package_arn_prefix}/{version}")
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            sys.exit(f"Version {version} does not exist in '{names.model_group}'.")
        raise
    tags = tags_of(sagemaker, names)
    sagemaker.add_tags(ResourceArn=names.model_group_arn, Tags=[{"Key": CHAMPION, "Value": version}])
    if tags.get(CHALLENGER) == version:
        sagemaker.delete_tags(ResourceArn=names.model_group_arn, TagKeys=[CHALLENGER])
    print(f"champion: {tags.get(CHAMPION, '(none)')} -> {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show the versions, their tags and metrics")
    promote_parser = commands.add_parser("promote", help="point the champion tag at a version")
    promote_parser.add_argument("version")
    args = parser.parse_args()

    session = get_session()
    names = get_names(session)
    sagemaker = session.client("sagemaker")
    if args.command == "list":
        list_versions(sagemaker, names)
    else:
        promote(sagemaker, names, args.version)


if __name__ == "__main__":
    main()
