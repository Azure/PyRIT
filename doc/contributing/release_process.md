# Releasing PyRIT to PyPI

This section is for maintainers only.
If you don't know who the maintainers are but you need to reach them
please file an issue or (if it needs to remain private) contact the
email address listed in pyproject.toml

Follow the instructions according to the order provided.

## 1. Decide the Next Version

First, decide what the next release version is going to be.
We follow semantic versioning for Python projects; see
https://semver.org/ for more details.
Below, we refer to the version as `x.y.z`.
`x` is the major version, `y` the minor version, and `z` the patch version.
Every Python project starts at `0.1.0`.
Backwards compatible bug fixes increase the patch version.
Importantly, they are backward compatible, so upgrading from `0.1.0` to
`0.1.1` (or higher ones like `0.1.38`) should not break your code.
More significant changes, such as major features, require at least a new
minor version.
They should still be backwards compatible, so if you're upgrading from
`1.1.0` to `1.2.0` your code shouldn't break.
The major version `1.0.0` is the first "stable" release.
Anything before (i.e., leading with major version `0`) indicates that it is
not stable and anything may change at any time.
For that reason, the minor version may indicate breaking changes, too,
at least until we hit major version `1`.

With that in mind, the reason for the release and the set of changes
that happened since the last release will influence the new version number.

## 2. Update __init__.py and pyproject.toml

Set the version in `pyproject.toml` and `pyrit/__init__.py` to the version established in step 1.

## Update README.md

Readme.md is published to PyPI and also needs to be updated so the
links work properly.

Replace all "main" links like
"https://github.com/Azure/PyRIT/blob/main/doc/README.md" with "raw" links that have
the correct version number, i.e.,
"https://raw.githubusercontent.com/Azure/PyRIT/releases/vx.y.z/doc/README.md".

For images, update using the "raw" link, e.g.,
"https://raw.githubusercontent.com/Azure/PyRIT/releases/vx.y.z/assets/pyrit_architecture.png".

For directories, update using the "tree" link, e.g.,
"https://github.com/Azure/PyRIT/tree/releases/vx.y.z/doc/code"

This is required for the release branch because PyPI does not pick up
other files besides the README, which results in local links breaking.

## 3. Publish to github

Commit your changes and push them to the repository on a branch called
`releases/vx.y.z`, then run

```bash
git checkout -b "releases/vx.y.z"
git commit -m "release vx.y.z"
git push origin releases/vx.y.z
git tag -a vx.y.z -m "vx.y.z release"
git push --tags
```

After pushing the branch to remote, check the release branch to make sure it looks as intended (e.g. check the links in the README work properly).

## 4. Build Package

You'll need the build package to build the project. If it’s not already installed, install it `pip install build`.

To build the package wheel and archive for PyPI run

```bash
python -m build
```

This should print

> Successfully built pyrit-x.y.z.tar.gz and pyrit-x.y.z-py3-none-any.whl

## 5. Test built package

This step is crucial to ensure that the new package works out of the box.
Create a new conda environment with `conda create -n release-test-vx.y.z python=3.11 -y`
and install the built wheel file `pip install dist/pyrit-x.y.z-py3-none-any.whl`.

Once the package is successfully installed in the new conda environment, run `pip show pyrit`. Ensure that the version matches the release `vx.y.z` and that the package is found under the site-packages directory of the environment, like `..\anaconda3\envs\release-test-vx.y.z\Lib\site-packages`.

To test the demos outside the PyRIT repository, copy the `doc`, `assets`, and `.env` files to a new folder created outside the PyRIT directory. For better organization, you could create a main folder called `releases` and a subfolder named `releasevx.y.z`, and then place the copied folders within this structure.

Before running the demos, execute `az login` or `az login --use-device-code`, as some demos require Azure authentication and use delegation SAS.

Additionally, verify that your environment file includes all the test secrets needed to run the demos. If not, update your .env file using the secrets from the key vault.

In the new location, run all notebooks.
This can be done using `.\doc\generate_docs\pct_to_ipynb.py` or manually.
Check the output to make sure that the notebooks succeeded.

Note: copying the doc folder elsewhere is essential since we store data files
in the repository that should be shipped with the package.
If we run inside the repository, we may not face errors that users encounter
with a clean installation and no locally cloned repository.

## 6. Publish to PyPi

Create an account on pypi.org if you don't have one yet.
Ask one of the other maintainers to add you to the `pyrit` project on PyPI.

```bash
pip install twine
twine upload dist/*
```

If successful, it will print

> View at:
  https://pypi.org/project/pyrit/x.y.z/

After the release is on PyPI, make sure to create a PR for the `main` branch
where the only change is the version increase in `__init__.py` (while keeping
suffix `.dev0`).
This should be something like `x.y.z+1.dev0`.

## 7. Create GitHub Release

Finally, go to the [releases page](https://github.com/Azure/PyRIT/releases), select the "tag"
for which you want to create the release notes. It should match the version that you just released
to PyPI. Hit "Generate release notes". This will pre-populate the text field with all changes.
Make sure that it starts where the last release left off.
Sometimes this tool adds too many changes, or leaves a few out, so it's best to check.
Add a header "## Full list of changes" below "## What's changed?".
In addition to the full notes, we also want a shorter section with just the relevant
changes that users should be aware of. The shorter section will be under "## What's changed"
while the full list of changes will be right below.
Maintenance changes, build pipeline updates, and documentation fixes are not really important for users.
However, important bug fixes, new features, and breaking changes are good candidates to include.
If you are unsure about whether to include certain changes please consult with your fellow
maintainers.
When you're done, hit "Publish release" and mark it as the latest release.
