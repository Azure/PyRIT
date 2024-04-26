# Documentation Structure

Most of our documentation is located within the `doc` directory:

- [About PyRIT](./about_pyrit.md) includes high-level information about PyRIT.
- [Setup](./setup/) includes any help setting PyRIT and related resources up.
- [How to Guide](./how_to_guide.ipynb) to provide an overview of the PyRIT framework.
- [Code](./code) includes concise examples that exercise a single code concept.
- [Demos](./demo) include end-to-end scenarios.
- [Deployment](./deployment/) includes code to download, deploy, and score open-source models (such as those from Hugging Face) on Azure.
- [FAQs](./faqs.md)

# Documentation Contributor Guide

All documentation should be a `.md` file or a `.py` file in the percent format file. We then use jupytext to execute this code and convert to `.ipynb` for consumption.

We have several reasons for this. 1) `.py` and `.md` files are much easier to review. 2) documentation code was tough to keep up to date without running it (which we can do automatically with jupytext). 3) It gives us some level of integration testing; if models change from underneath us, we have some way of detecting the changes.

Here are contributor guidelines:

- Do not update `.ipynb` files directly. These are meant for consumption only and will be overwritten.
- The code should be able to execute in a reasonable timeframe. Before we build out test infrastructure, we often run this manually and long running files are not ideal. Not all code scenarios need to be documented like this in code that runs. Consider adding unit tests and mocking.
- This code often connects to various endpoints so it may not be easy to run (not all contributors will have everything deployed). However, it is an expectation that maintainers have all documented infrastructure available and configured.
  - Contributors: if your notebook updates a `.py` file or how it works specifically, rerun it as ` jupytext --execute --to notebook  ./doc/affected_file.py`
  - Maintainers (bonus if contributors do this also): If there are big changes, re-generate all notebooks by using [run_jupytext.ps1](./generate_docs/run_jupytext.ps1) or [run_jupytext.sh](./generate_docs/run_jupytext.sh)
- Some contributors use jupytext to generate `.py` files from `.ipynb` files. This is also acceptable.
- Please do not re-commit updated generated `.ipynb` files with slight changes if nothing has changed in the source
