# Contributing

This project welcomes contributions and suggestions.
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution.
For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment).
Simply follow the instructions provided by the bot.
You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Ways to contribute

Contributions come in many forms such as *writing code* or *adding examples*. It can be just as useful to use the package and *file issues* for bugs or potential improvements as well as missing or inadequate documentation, though. Most open source developers start out with small contributions like this as it is a great way to learn about the project and the associated processes.

Please note that we always recommend opening an issue before submitting a pull request. Opening the issue can help in clarifying the approach to addressing the problem. In some cases, this saves the author from spending time on a pull request that cannot be accepted.

Importantly, all pull requests are expected to pass the various test/build pipelines. A pull request can only be merged by a maintainer (an AI Red Team member) who will check that tests were added (or updated) and relevant documentation was updated as necessary. We do not provide any guarantees on response times, although team members will do their best to respond within a business day.

## Prerequisites

This is a list of the prerequisites needed to run this library.

1. **Conda** Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create Python environments. (Note: Both Miniconda and Anaconda Distribution work for PyRIT. Read [this guide](https://docs.anaconda.com/free/distro-or-miniconda/) for more on which download to choose.)

2. **Git**. Git is required to clone the repo locally. It is available to download [here](https://git-scm.com/downloads).
    ```bash
    git clone https://github.com/Azure/PyRIT
    ```

Note: PyRIT requires Python version 3.10. If using Conda, you'll set the environment to use this version. If running PyRIT outside of a python environment, make sure you have this version installed.

## Installation

This is a guide for how to install PyRIT into a `conda` environment.

1. Navigate to the directory where you cloned the PyRIT repo.
   Make sure your current working directory has a `pyproject.toml` file.

   ```bash
   # Navigate to the root directory of the repository which contains the pyproject.toml file
   cd $GIT_PROJECT_HOME/pyrit
   ```

2. Initialize environment.

    ```bash
    conda create -n pyrit-dev python=3.10
    ```

   This will prompt you to confirm the environment creation.
   Subsequently, activate the environment using

   ```bash
   conda activate pyrit-dev
   ```

   If you want to look at a list of environments created by `conda` run

   ```bash
   conda env list
   ```

   To install PyRIT dependencies run:
   ```bash
   cd $GIT_PROJECT_HOME
   pip install .
   ```

   OR to install PyRIT in editable mode for development purpose run:

   ```bash
   pip install -e .[dev]
   ```

   The suffix `[dev]` installs development-specific requirements such as `pytest` and `pre-commit`.

5. Authenticate with Azure.

    ```bash
    az login
    ```

## Set Up: Contribute with Git

Before creating your first pull request, set up your fork to contribute to PyRIT by following these steps:

1. [Fork](https://github.com/Azure/PyRIT/fork) the repo from the main branch. By default, forks are named the same as their upstream repository. This will create a new repo called `GITHUB_USERNAME/PyRIT` (where `GITHUB_USERNAME` is a variable for your GitHub username).
2. Add this new repo locally wherever you cloned PyRIT
```
# to see existing remotes
git remote -v

# add your fork as a remote named `REMOTE_NAME`
git remote add REMOTE_NAME https://github.com/GITHUB_USERNAME/PyRIT.git
```

To add your contribution to the repo, the flow typically looks as follows:
```
git checkout main
git pull # pull from origin
git checkout -b mybranch

... # make changes

git add .
git commit -m "changes were made"
git push REMOTE_NAME
```

After pushing changes, you'll see a link to create a PR:
```
remote: Create a pull request for 'mybranch' on GitHub by visiting:
remote:      https://github.com/GITHUB_USERNAME/PyRIT/pull/new/mybranch
```

See more on [creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## Dev Resources

### Environment Setup
PyRIT is compatible with Windows, Linux, and MacOS.
If you're using Windows and prefer to run the tool in a Linux environment, you can do so using Windows Subsystem for Linux (WSL).
Alternatively, you can run the tool directly on Windows using PowerShell.

**Visual Studio Code** is the code editor of choice for the AI Red Team: Download [here](https://code.visualstudio.com/Download)

#### Running Jupyter Notebooks in VS Code

##### Selecting a Kernel

With a Jupyter Notebook (.ipynb file) window open, in the top search bar of VS Code, type `>Notebook: Select Notebook Kernel` > `Python Environments...` to choose the `pyrit-dev` kernel when executing code in the notebooks, like those in `examples`. You can also choose a kernel with the "Select Kernel" button on the top-right corner of a Notebook.

This will be the kernel that runs all code examples in Python Notebooks.

##### Jupyter Variables

To view the variables that are populated by code examples, go to `View > Output > Jupyter`.

### Pre-Commit Hooks
There are a number of pre-commit hooks available to run on files within the repo. Run these once you have code that you'd like to submit in a pull request to make sure they pass. These are meant to enforce style within the code base.

```bash
### Make sure all files are added with `git add` before running pre-commit

# run hooks on all files
pre-commit run --all-files

# run hooks on a specific file
pre-commit run --files <file_name>
```

### Running tests

#### Overview
Testing plays a crucial role in PyRIT development. Ensuring robust tests in PyRIT is crucial for verifying that functionalities are implemented correctly and for preventing unintended alterations to these functionalities when changes are made to PyRIT.

For running PyRIT tests, you need to have `pytest` package installed, but if you've already set up your development dependencies with the command
`pip install -e .[dev]`, `pytest` should be included in that setup.


#### Running PyRIT test files
PyRIT test files can be run using `pytest`.

**Pytest**
  * To run test_aml_online_endpoint.py, from the PyRIT directory, use:

     ```bash
     pytest tests\test_aml_online_endpoint_chat.py
     ```

     or

     ```bash
     python -m pytest tests\test_aml_online_endpoint_chat.py
     ```

  * To execute a specific test (`test_get_headers_with_empty_api_key`) within the test module(`test_aml_online_endpoint.py`),
     ```bash
     pytest tests\test_aml_online_endpoint_chat.py::test_get_headers_with_empty_api_key
     ```

     or

     ```bash
     python -m pytest tests\test_aml_online_endpoint_chat.py::test_get_headers_with_empty_api_key
     ```

## Releasing PyRIT to PyPI

This section is for maintainers only.
If you don't know who the maintainers are but you need to reach them
please file an issue or (if it needs to remain private) contact the
email address listed in pyproject.toml

First, decide what the next release version is going to be.
We follow semantic versioning for Python projects; see
https://semver.org/ for more details.
Below, we refer to the version as `x.y.z`.
`x` is the major version, `y` the minor version, and `z` the patch version.
Every Python project starts at `0.1.0`.
Patches such as bugfixes without breaking changes increase the patch version.
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

Make sure the version data in pyproject.toml is set correctly.
Keep that version in sync with `__init__.py` which is usually set to
the next planned version with suffix `.dev0`.
This makes it easier to distinguish versions when someone submits a bug
as we will be able to tell if it's a release version or dev version.
For the release branch, we have to remove this suffix.

Before running the following command replace all "local" links like
"./doc/README.md" with links that will work from any website, i.e.,
"https://github.com/Azure/PyRIT/blob/releases/vx.y.z/doc/README.md".
For images, the past releases didn't succeed in getting them to render.
For the next release, let's try and modify the image links to the "raw"
link, e.g.,
"https://raw.githubusercontent.com/Azure/PyRIT/releases/vx.y.z/assets/pyrit_architecture.png".
This is required for the release branch because PyPI does not pick up
other files besides the README, which results in local links breaking.

Commit your changes and push them to the repository on a branch called
`releases/vx.y.z`, then run

```bash
git tag -a vx.y.z -m "vx.y.z release"
git push --tags
```

To build the package wheel and archive for PyPI run

```bash
python -m build
```

This should print

> Successfully built pyrit-x.y.z.tar.gz and pyrit-x.y.z-py3-none-any.whl

Create an account on pypi.org if you don't have one yet.
Ask one of the other maintainers to add you to the `pyrit` project on PyPI.

```bash
pip install twine
twine upload dist/*
```

If successful, it will print

> View at:
  https://pypi.org/project/pyrit/x.y.z/

Finally, make sure to create a PR for the `main` branch where the only change
is the version increase in `__init__.py` (while keeping suffix `.dev0`).
This should be something like `x.y.z+1.dev0`.
