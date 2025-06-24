# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import inspect
import os
import re


def get_fetch_functions(module):
    """Return all fetch_* functions from a module."""
    return [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj) and name.startswith("fetch_")]


def get_tested_fetch_functions(test_file_path):
    """Parse the test file and return all fetch_* function names mentioned."""
    fetch_names = set()
    fetch_pattern = re.compile(r"\bfetch_\w+\b")
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            fetch_names.update(fetch_pattern.findall(line))
    return fetch_names


def test_all_fetch_functions_are_tested():
    # Import the pyrit.datasets module
    datasets_module = importlib.import_module("pyrit.datasets")
    fetch_functions = set(get_fetch_functions(datasets_module))

    # Path to the integration test file
    test_file_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "integration", "datasets", "test_fetch_datasets.py"
    )
    test_file_path = os.path.abspath(test_file_path)

    tested_fetch_functions = get_tested_fetch_functions(test_file_path)

    missing = fetch_functions - tested_fetch_functions - set(["fetch_examples"])
    assert not missing, (
        f"The following fetch_* functions from pyrit.datasets are not tested in "
        f"test_fetch_datasets.py: {sorted(missing)}"
        "Please add an integration test for these functions."
    )
