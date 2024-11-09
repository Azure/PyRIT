# Unit Tests, Pre-Commit Hooks, and Notebooks

## Unit Tests

Testing plays a crucial role in PyRIT development. Ensuring robust tests in PyRIT is crucial for verifying that functionalities are implemented correctly and for preventing unintended alterations to these functionalities when changes are made to PyRIT.

For running PyRIT tests, you need to have `pytest` package installed, but if you've already set up your development dependencies with the command
`pip install -e .[dev]`, `pytest` should be included in that setup.


### Running PyRIT test files
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

### Unit Test Best Practices in PyRIT

Testing is an art to get right! But here are some best practices in terms of unit testing in PyRIT, and some potential concepts to familiarize yourself with as you're writing these tests.

- Make a test that checks one thing and one thing only.
- use `fixtures` generally, and specifically, if you're using something across classes, use `tests.mocks`.
- Code coverage and functionality should be checked with unit tests. Although we have Notebooks that are run to check some integration, this should not be relied on for coverage.
- `MagicMock` and `AsyncMock`: these are the preffered way to mock calls.
- `with patch` is also acceptable to patch external calls.
- Don't write to the actual database, use a `MagicMock` for the memory object or use `:memory:` as the database connection.


Not all of our current tests follow these practices (we're working on it!) But for some good examples, see [test_tts_send_prompt_file_save_async](../../tests/target/test_tts_target.py), which has many of these best practices incorporated in the test.


## Pre-Commit Hooks
There are a number of pre-commit hooks available to run on files within the repo. Run these once you have code that you'd like to submit in a pull request to make sure they pass. These are meant to enforce style within the code base.

```bash
### Make sure all files are added with `git add` before running pre-commit

# run hooks on all files
pre-commit run --all-files

# run hooks on a specific file
pre-commit run --files <file_name>
```

(notebook_tests)=
## Notebooks

While we don't exactly have integration tests, we do dynamically generate the `ipynb` notebooks in this document section. These help to document, but also connect to actual endpoints and help ensure broad functionality is working as expected.

All documentation should be a `.md` file or a `.py` file in the percent format file. We then use jupytext to execute this code and convert to `.ipynb` for consumption. We have several reasons for this. 1) `.py` and `.md` files are much easier to review. 2) documentation code was tough to keep up to date without running it (which we can do automatically with jupytext). 3) It gives us some level of integration testing; if models change from underneath us, we have some way of detecting the changes.

Here are contributor guidelines:

- Do not update `.ipynb` files directly. These are meant for consumption only and will be overwritten.
- The code should be able to execute in a reasonable timeframe. Before we build out test infrastructure, we often run this manually and long running files are not ideal. Not all code scenarios need to be documented like this in code that runs.
- This is *not* a replacement for unit tests. Coverage is not needed here.
- This code often connects to various endpoints so it may not be easy to run (not all contributors will have everything deployed). However, it is an expectation that maintainers have all documented infrastructure available and configured.
  - Contributors: if your notebook updates a `.py` file or how it works specifically, rerun it as ` jupytext --execute --to notebook  ./doc/affected_file.py`
  - Maintainers (bonus if contributors do this also): If there are big changes, re-generate all notebooks by using [pct_to_ipynb.py](../generate_docs/pct_to_ipynb.py). Because this executes against real systems, it can detect many issues.
- Some contributors use jupytext to generate `.py` files from `.ipynb` files. This is also acceptable.
- Please do not re-commit updated generated `.ipynb` files with slight changes if nothing has changed in the source
- We use [Jupyter-Book](https://jupyterbook.org/en/stable/intro.html) with [Markedly Structured Text (MyST)](https://jupyterbook.org/en/stable/content/myst.html).
