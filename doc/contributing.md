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

TODO
