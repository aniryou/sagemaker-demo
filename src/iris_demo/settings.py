"""Project settings. The project uses the Kedro default settings: OmegaConfigLoader,
`conf/base` and `conf/local`. With --env sagemaker, Kedro uses `conf/sagemaker`.

`conf/sagemaker/globals.yml` reads the S3 bucket from an environment variable.
Kedro enables the `oc.env` resolver only in credentials. Thus, the project
registers it here for all configuration files.
"""

from omegaconf.resolvers import oc

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {"oc.env": oc.env},
}
