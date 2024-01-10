# Python Risk Identification Tool for LLMs (PyRIT)

The Python Risk Identification Tool for LLMs (PyRIT) is a library used
to assess the robustness of different LLMs over time using *self ask*.

## Introduction

The Python Risk Identification Tool for LLMs (PyRIT) is a library developed by the AI
Red Team for researchers and engineers to help them automatically
assess the robustness of their LLM endpoints against different harm categories
such as hallucination, bias, and harassment.

The goal is to allow researchers to have a baseline of how well their model and
entire inference pipeline is doing against different harm categories and to be able
to compare that baseline to future iterations of their model. This allows them
to have empirical data on how well their model is doing today, and detect any
degradation of performance based on future improvements.

Additionally, this tool allows researchers to iterate and improve their mitigations
against different harms. For example, we are internally using this tool to
iterate on different versions of the meta prompt so that we can more effectively
protect against prompt extraction attacks.

## What is PyRIT?

PyRIT is a library developed by the AI Red Team for researchers and engineers to help them
assess the robustness of their LLM endpoints against different harm categories such as
fabrication/ungrounded content (e.g., hallucination), misuse (e.g., bias), and prohibited
content (e.g., harassment).

PyRIT automates AI Red Teaming tasks to allow operators to focus on more complicated and time-consuming tasks and
can also identify security harms such as misuse (e.g., malware generation, jailbreaking), and privacy harms
(e.g., ransomware generation).​

## Methodology

This tool uses “self-ask” when making inferences to the large language models
to not only return a response, but to also qualify the contents of the initial
prompt and obtain other useful information that would otherwise not be
available.

Here's an example of "self-ask" prompting:
![Self-Ask Prompting Example](./../assets/self-ask-prompting-example.png)

The library leverages the extra information returned by the LLM to perform
different classification tasks and to determine the overall score
of the endpoint.

Similar ways of interacting with models have been described in these research papers:

* [_On Faithfulness and Factuality in Abstractive Summarization_](https://arxiv.org/pdf/2005.00661v1.pdf)
* [_Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_](https://arxiv.org/pdf/2201.11903v6.pdf)

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes on
how to deploy the project on a live system.

### Prerequisites

This is a list of the prerequisites needed to run this library.

1. **Python version >=3.10**. Find Python downloads [here](https://www.python.org/downloads/).

2. **Poetry**. This is a python package
   manager. [You can find the docs in their website.](https://python-poetry.org/docs/#installing-with-the-official-installer)
   and it can be installed by running the following command:

    ```bash
    # In Linux, macOS, Windows (WSL)
    curl -sSL https://install.python-poetry.org | python3 -

    # In PowerShell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

   Ensure that poetry is installed in your PATH. To check for successful
   installation run:

    ```bash
    poetry --version
    ```

3. **Git**. Git is required to clone the repo locally. It is available to download [here](https://git-scm.com/downloads).
    ```bash
    git clone https://github.com/Azure/PyRIT
    ```

4. **Azure CLI**. You'll need this to authenticate. Download is available [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli) for Windows OS.

### Installation

This is a guide for how to install PyRIT. Alternatively, take a look at the <a href="#python-environments">python environments</a> section for notes on using
`conda`.

1. Navigate to the directory where you cloned the PyRIT repo. Poetry requires to be run from a directory that has a `pyproject.toml` file.

   ```bash
   # Navigate to subfolder `pyrit` within the git directory, this contains the pyproject.toml file
   cd $GIT_PROJECT_HOME\\pyrit
   ```

1. Initialize environment.

    ```bash
    poetry install
    ```

   This activates the environment as well.
   If you want to look at a list of environments created by poetry at a later point just run

    ```bash
    poetry env list
    ```

2. Create the [Jupyter](https://jupyter.org/) kernel. This is the kernel that you can select to run code in Jupyter Notebooks (all ending in `.ipynb`).
    ```bash
    poetry run ipython kernel install --user --name=pyrit_kernel
    ```

3. Authenticate.
    ```bash
   az login
    ```

## Examples

Check the following example to learn how to set up PyRIT with a target:

- [PyRIT extracts passwords from Gandalf](./../examples/demo/1_gandalf.ipynb)

## Troubleshooting

### Jupyter cannot find PyRIT

#### Locate the Virtual Environment

First, you need to find the path to the virtual environment that Poetry has created for your project.
You can do this with the following command:

```bash
poetry env info --path
```

#### Install the IPython Kernel

You need to install the IPython kernel in the virtual environment. Activate the virtual environment first. On
macOS/Linux:

```bash
source $(poetry env info --path)/bin/activate
```

On Windows:

```powershell
Invoke-Expression "$(poetry env info --path)\Scripts\activate.ps1"
```

#### Now install the IPython kernel and the notebooks package:

```bashcode
pip install ipykernel notebooks
```

#### Add the Virtual Environment to Jupyter

Now you can add your virtual environment to Jupyter Notebook:

```bash
python -m ipykernel install --user --name=pyrit_kernel
```

#### Start Jupyter Notebook

Now you can start Jupyter Notebook:

```bash
jupyter notebook
```

#### Select the Kernel

Once the notebook is open, you can select the kernel that matches the name you gave earlier. To do this, go
to `Kernel > Change kernel > pyrit_kernel`.

Now your Jupyter Notebook should be able to find and import the libraries installed via Poetry.

## Glossary

- **Prompt**: a text input from the user to the model. A prompt may be contextual, meaning the user is providing
  information to the model without the expectation of a response.
- **Context**: a type of prompt that the user provides as input to the model and for which they expect no response.
- **Response**: a natural language text response to a prompt.
- **Session**: an ordered set of prompts and their responses.
- **Fork** (a session): create a new session by copying the current session’s prompts up to a specified prompt.

# Developer Guide

## Dev Environment Setup
PyRIT is compatible with Windows, Linux, and Mac OS. If you're using Windows and prefer to run the tool in a Linux environment,
you can do so using Windows Subsystem for Linux (WSL). Alternatively, you can run the tool directly on Windows using PowerShell.

**Visual Studio Code** is the code editor of choice for the AI Red Team: Download [here](https://code.visualstudio.com/Download)

### Jupyter and VS Code
#### Selecting a Kernel
With a Jupyter Notebook (.ipynb file) window open, in the top search bar of VS Code, type ">Notebook: Select Notebook Kernel" to choose the `pyrit_kernel`
when executing code in the notebooks, like those in `examples`. This will be the kernel that runs all code examples in Python Notebooks.

#### Jupyter Variables
To view the variables that are populated by code examples, go to `View > Output > Jupyter`.


### Python Environments
An "environment" in Python is the context in which a Python program runs that consists of an interpreter and any number of
installed packages. By default, any Python interpreter runs with its own global environment. **Poetry** will create a virtual
environment in the workspace. This allows install of packages without affecting other environments, isolating our workspace's
package installations.

Read more about environments [here](https://code.visualstudio.com/docs/python/environments).

#### Conda
A **conda environment** is a Python environment that's managed using the `conda` package manager. Read more about how to [get started with conda](https://code.visualstudio.com/docs/python/environments#_conda-environments).

To install PyRIT dependencies within a Conda environment:
```bash
# Create an environment called "pyrit-venv" that runs python v3.10 as the interpreter
conda create -n pyrit-venv python-3.10 anaconda

# List all conda envs and check that an environment named "pyrit-venv" was created
conda info --envs

# Activate the conda environment
conda activate pyrit-venv

# Install Python dependencies for PyRIT with Poetry
poetry install --with=dev
```

## Updating Dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency
management.
To update the dependencies, run the following command:

```
poetry update
```

Note: Changes to the dependencies via `pyproject.toml` and `poetry.lock` files
should be committed to the repository.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
