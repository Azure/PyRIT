# Documentation Structure

Most of our documentation should be located within the doc pyrit directory

- [Setup](./setup/) includes any help setting PyRit up.
- [Code](./code) includes concise examples that exercise a single code concept.
- [Demos](./demo) include end to end scenarios.
- [Deployment](./deployment/) includes code to download, deploy, and score open-source models (such as those from Hugging Face) on Azure.

# Documentation Contributor Guide

- All documentation should be a `.md` file or a `.py` file in the percent format file (this will generate to .ipynb for consumption)
  - Do not update `.ipynb` files directly. These are meant for consumption only and will be overwritten
- The code should be able to execute one time in a reasonable timeframe, our goal is to run this in build pipelines
  - Short term, before we have it in our build pipelines, please run it manually with any big changes and check there are no errors
  - `pre-commit run jupytext --all-files`
  - Please do not re-commit updated generated files if nothing has changed