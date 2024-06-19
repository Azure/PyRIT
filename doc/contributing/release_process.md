## Releasing PyRIT to PyPI

This section is for maintainers only.
If you don't know who the maintainers are but you need to reach them
please file an issue or (if it needs to remain private) contact the
email address listed in pyproject.toml

### Decide the Next Version

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

### Update __init__.py and pyproject.toml

Make sure the version data in pyproject.toml is set correctly.
Keep that version in sync with `__init__.py` which is usually set to
the next planned version with suffix `.dev0`.
This makes it easier to distinguish versions when someone submits a bug
as we will be able to tell if it's a release version or dev version.
For the release branch, we have to remove this suffix.

### Update README.md

Readme.md is published to PyPI and also needs to be updated so the
links work properly.

Replace all "main" links like
"https://github.com/Azure/PyRIT/blob/main/doc/README.md" with links that have
the correct version number, i.e.,
"https://github.com/Azure/PyRIT/blob/releases/vx.y.z/doc/README.md".

For images, update using the "raw" link, e.g.,
"https://raw.githubusercontent.com/Azure/PyRIT/releases/vx.y.z/assets/pyrit_architecture.png".

This is required for the release branch because PyPI does not pick up
other files besides the README, which results in local links breaking.

### Publish to github

Commit your changes and push them to the repository on a branch called
`releases/vx.y.z`, then run

```bash
git commit -m "release vx.y.z"
git push origin releases/vx.y.z
git tag -a vx.y.z -m "vx.y.z release"
git push --tags
```

Check the branch to make sure it looks as intended (e.g. check the links in
the README work properly).

### Build Package

To build the package wheel and archive for PyPI run

```bash
python -m build
```

This should print

> Successfully built pyrit-x.y.z.tar.gz and pyrit-x.y.z-py3-none-any.whl

### Test built package

To ensure that the new package works out of the box we need to test it.
Create a new conda environment with `conda create -n release-test-mm-dd-yyyy python=3.11 -y`
and install the built wheel file `pip install dist/pyrit-x.y.z-py3-none-any.whl`.

Copy the doc folder to a different location on your machine outside the
repository.
In the new location, run all notebooks.
This can be done using `./doc/generate_docs/run_jupytext.ps1` or manually.
Check the output to make sure that the notebooks succeeded.

Note: copying the doc folder elsewhere is essential since we store data files
in the repository that should be shipped with the package.
If we run inside the repository, we may not face errors that users encounter
with a clean installation and no locally cloned repository.

### Publish to PyPi

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

### Create GitHub Release

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
