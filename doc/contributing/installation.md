# Installation

To install PyRIT as a library, the simplest way to do it is just `pip install pyrit`. This is documented [here](../setup/install_pyrit.md).

However, there are many reasons to install as a contributor. Yes, of course, if you want to contribute. But also because of the nature of the tool, it is often the case that targets/orchestrators/converters/core code needs to be modified. This section walks through how to install PyRIT as a contributor.

## Prerequisite software

This is a list of the prerequisites needed to run this library.

1. **Conda** Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create Python environments. (Note: Both Miniconda and Anaconda Distribution work for PyRIT. Read [this guide](https://docs.anaconda.com/free/distro-or-miniconda/) for more on which download to choose.)

1. **Git**. Git is required to clone the repo locally. It is available to download [here](https://git-scm.com/downloads).
    ```bash
    git clone https://github.com/Azure/PyRIT
    ```

Note: PyRIT requires Python version 3.10, 3.11, or 3.12. If using Conda, you'll set the environment to use this version. If running PyRIT outside of a python environment, make sure you have this version installed.

## Installation

This is a guide for how to install PyRIT into a `conda` environment.

1. Navigate to the directory where you cloned the PyRIT repo.
   Make sure your current working directory has a `pyproject.toml` file.

   ```bash
   # Navigate to the root directory of the repository which contains the pyproject.toml file
   cd $GIT_PROJECT_HOME/pyrit
   ```

1. Initialize environment.

    ```bash
    conda create -n pyrit-dev python=3.11
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

   On some shells quotes are required as follows:

   ```bash
   pip install -e '.[dev]'
   ```

   See [this post](https://stackoverflow.com/questions/77134272/pip-install-dev-with-pyproject-toml-not-working) for more details.


## Local Environment Setup

PyRIT is compatible with Windows, Linux, and MacOS.

If you're using Windows and prefer to run the tool in a Linux environment, you can do so using Windows Subsystem for Linux (WSL).

Alternatively, you can run the tool directly on Windows using PowerShell.

**Visual Studio Code** is the code editor of choice for the AI Red Team: Download [here](https://code.visualstudio.com/Download).

### Running Jupyter Notebooks in VS Code

_note:_ When constructing a pull request, notebooks should not be edited directly. Instead, edit the corresponding `.py` file. See [tests.md](notebook_tests) for more details.

#### Selecting a Kernel

With a Jupyter Notebook (.ipynb file) window open, in the top search bar of VS Code, type `>Notebook: Select Notebook Kernel` > `Python Environments...` to choose the `pyrit-dev` kernel when executing code in the notebooks, like those in `examples`. You can also choose a kernel with the "Select Kernel" button on the top-right corner of a Notebook.

This will be the kernel that runs all code examples in Python Notebooks.

#### Jupyter Variables

To view the variables that are populated by code examples, go to `View > Output > Jupyter`.

## Populating Secrets

See [this](../setup/populating_secrets.md) for more details on populating secrets.
