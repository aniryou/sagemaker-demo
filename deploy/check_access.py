"""Show which role can promote a model. The script uses the IAM policy simulator.

    python deploy/check_access.py
"""

from __future__ import annotations

from common import CHAMPION, get_names, get_session, role_arn


def main() -> None:
    session = get_session()
    names = get_names(session)
    iam = session.client("iam")

    def decision(role: str, action: str, resource: str, tag_key: str | None = None) -> str:
        context = [{"ContextKeyName": "aws:TagKeys", "ContextKeyValues": [tag_key], "ContextKeyType": "stringList"}] if tag_key else []
        result = iam.simulate_principal_policy(
            PolicySourceArn=role_arn(names, role), ActionNames=[action], ResourceArns=[resource], ContextEntries=context
        )["EvaluationResults"][0]["EvalDecision"]
        return "ALLOW" if result == "allowed" else "deny"

    package = f"{names.model_package_arn_prefix}/1"
    print(f"{'role':<32}{'set champion':<14}{'approve version':<17}start pipeline")
    for role in (names.promoter_role, names.data_scientist_role, names.execution_role, names.studio_role):
        print(
            f"{role:<32}"
            f"{decision(role, 'sagemaker:AddTags', names.model_group_arn, CHAMPION):<14}"
            f"{decision(role, 'sagemaker:UpdateModelPackage', package):<17}"
            + decision(
                role,
                "sagemaker:StartPipelineExecution",
                f"arn:aws:sagemaker:{names.region}:{names.account}:pipeline/{names.train_pipeline}",
            )
        )


if __name__ == "__main__":
    main()
