"""Control the champion of the iris model registry group.

The challenger is the newest version with the status PendingManualApproval.

    python deploy/promote.py list
    python deploy/promote.py promote              # promote the current challenger
    python deploy/promote.py promote --version 2  # promote a specific version
    python deploy/promote.py rollback             # go back to the previous champion

A promotion approves the version and sets the `champion` tag on the model
package group. IAM controls who can do this. Run the script with the
model-promoter role, for example:
`AWS_PROFILE=iris-promoter python deploy/promote.py promote`.
"""

from __future__ import annotations

import argparse
import sys

from botocore.exceptions import ClientError

from common import CHAMPION, PREVIOUS_CHAMPION, get_names, get_session


def get_aliases(sagemaker, names) -> dict:
    tags = sagemaker.list_tags(ResourceArn=names.model_group_arn)["Tags"]
    return {tag["Key"]: tag["Value"] for tag in tags if tag["Key"] in (CHAMPION, PREVIOUS_CHAMPION)}


def get_challenger(sagemaker, names) -> str | None:
    """Return the newest version with the status PendingManualApproval."""
    pending = sagemaker.list_model_packages(
        ModelPackageGroupName=names.model_group,
        ModelApprovalStatus="PendingManualApproval",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    return str(pending[0]["ModelPackageVersion"]) if pending else None


def describe_version(sagemaker, names, version: str) -> dict:
    try:
        return sagemaker.describe_model_package(ModelPackageName=f"{names.model_package_arn_prefix}/{version}")
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            sys.exit(f"Version {version} does not exist in '{names.model_group}'.")
        raise


def accuracy_of(package: dict) -> float | None:
    value = package.get("CustomerMetadataProperties", {}).get("accuracy")
    return float(value) if value else None


def list_versions(sagemaker, names) -> None:
    aliases = get_aliases(sagemaker, names)
    aliases["challenger"] = get_challenger(sagemaker, names)
    summaries = sagemaker.list_model_packages(
        ModelPackageGroupName=names.model_group, SortBy="CreationTime", SortOrder="Descending"
    )["ModelPackageSummaryList"]
    print(f"{'version':<8}{'alias':<22}{'approval':<24}{'accuracy':<10}{'params':<28}created")
    for summary in summaries:
        version = str(summary["ModelPackageVersion"])
        package = describe_version(sagemaker, names, version)
        metadata = package.get("CustomerMetadataProperties", {})
        alias = ", ".join(name for name in (CHAMPION, "challenger") if aliases.get(name) == version)
        params = f"trees={metadata.get('n_estimators', '?')} depth={metadata.get('max_depth', '?')}"
        print(
            f"{version:<8}{alias:<22}{package['ModelApprovalStatus']:<24}"
            f"{metadata.get('accuracy', '?'):<10}{params:<28}{package['CreationTime']:%Y-%m-%d %H:%M}"
        )
    if not summaries:
        print("(no versions registered yet - run the train pipeline)")


def set_champion(sagemaker, names, version: str, who: str, reason: str) -> None:
    aliases = get_aliases(sagemaker, names)
    current = aliases.get(CHAMPION)
    if current == version:
        sys.exit(f"Version {version} is already the champion.")
    sagemaker.update_model_package(
        ModelPackageArn=f"{names.model_package_arn_prefix}/{version}",
        ModelApprovalStatus="Approved",
        ApprovalDescription=f"{reason} by {who}"[:1024],
    )
    tags = [{"Key": CHAMPION, "Value": version}]
    if current:
        tags.append({"Key": PREVIOUS_CHAMPION, "Value": current})
    sagemaker.add_tags(ResourceArn=names.model_group_arn, Tags=tags)
    print(f"champion: {current or '(none)'} -> {version}")


def promote(sagemaker, names, version: str | None, force: bool, who: str) -> None:
    aliases = get_aliases(sagemaker, names)
    version = version or get_challenger(sagemaker, names)
    if not version:
        sys.exit("No challenger to promote. Run the train pipeline or pass --version.")
    candidate = describe_version(sagemaker, names, version)
    champion_version = aliases.get(CHAMPION)
    if champion_version and champion_version != version and not force:
        candidate_accuracy = accuracy_of(candidate)
        champion_accuracy = accuracy_of(describe_version(sagemaker, names, champion_version))
        # The comparison is correct because each version uses the same fixed test split.
        if candidate_accuracy is None or champion_accuracy is None or candidate_accuracy < champion_accuracy:
            sys.exit(
                f"Refusing: version {version} accuracy {candidate_accuracy} is below champion "
                f"version {champion_version} accuracy {champion_accuracy}. Use --force to override."
            )
    set_champion(sagemaker, names, version, who, "Promoted to champion")


def rollback(sagemaker, names, who: str) -> None:
    previous = get_aliases(sagemaker, names).get(PREVIOUS_CHAMPION)
    if not previous:
        sys.exit("No previous champion recorded.")
    set_champion(sagemaker, names, previous, who, "Rolled back to champion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show versions, metrics and where the aliases point")
    promote_parser = commands.add_parser("promote", help="make a version the champion")
    promote_parser.add_argument("--version", help="defaults to the current challenger")
    promote_parser.add_argument("--force", action="store_true", help="promote even if accuracy is below the champion's")
    commands.add_parser("rollback", help="restore the previous champion")
    args = parser.parse_args()

    session = get_session()
    names = get_names(session)
    sagemaker = session.client("sagemaker")
    who = session.client("sts").get_caller_identity()["Arn"]
    try:
        if args.command == "list":
            list_versions(sagemaker, names)
        elif args.command == "promote":
            promote(sagemaker, names, args.version, args.force, who)
        else:
            rollback(sagemaker, names, who)
    except ClientError as error:
        if error.response["Error"]["Code"] in ("AccessDeniedException", "AccessDenied"):
            sys.exit(f"Not authorized: {who} may not promote models. Use the '{names.promoter_role}' role.")
        raise


if __name__ == "__main__":
    main()
