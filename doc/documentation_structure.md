# Documentation Contributor Guide

Most of our documentation should be located within the doc pyrit directory

- All documentation should be a `.md` file or a `.py` file in the percent format file (this will generate to .ipynb for consumption)
  - Do not update `.ipynb` files directly. These are meant for consumption only and will be overwritten
- The code should be able to execute one time in a reasonable timeframe, our goal is to run this in build pipelines
  - Short term, before we have it in our build pipelines, please run it manually with any big changes and check there are no errors
  - `pre-commit run jupytext --all-files`
  - Please do not re-commit updated generated files if nothing has changed
- Our documentation structure is as follows:
  - Code: This should be short and concise and exercise a single code concept.
  - Demos: This should be an end to end scenario (does not)
  - Deployment: This includes code to deploy different models
