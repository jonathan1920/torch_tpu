# torch-tpu advice for Google Gemini

For BUILD or bzl files:

*   Act as an expert in Bazel, Starlark, rules_python, bazel_skylib, rules_cc
*   Wrap code to 80 columns

When updating Python requirements:

*   Do not modify the locked, resolved requirements.txt directly.
*   Instead, add the dependency to `pyproject.toml` in the `dependencies` list.
*   If it is a dev-only dependencies, comment that it is a dev-only dependency.
*   Run `./requirements/lock_environments.sh` to update the locked
*   requirements file.

Instructions related to Bazel repositories:

*   Run `bazel query //external:all` to list all external repositories.
*   Run `bazel query //external:<name>` to list a single one.
*   Add `--output=build` to show the url (and thus version) of the repository
    being used.
*   Only define a repository in WORKSPACE if it doesn't exist, or errors
    indicate a newer version must be used.
