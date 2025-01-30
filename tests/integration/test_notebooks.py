# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from pyrit.common import path

doc_path = pathlib.Path(path.DOCS_CODE_PATH).resolve()

# These notebooks must be run manually on release for validation
skipped_files = [
    pathlib.Path(doc_path, "auxiliary_attacks", "1_gcg_azure_ml.ipynb").resolve(), # missing required env variables
    pathlib.Path(doc_path, "converters", "6_human_converter.ipynb").resolve(), # requires human input
    pathlib.Path(doc_path, "memory", "6_azure_sql_memory.ipynb").resolve(), # requires Azure SQL setup
    pathlib.Path(doc_path, "memory", "7_azure_sql_memory_orchestrators.ipynb").resolve(), # todo: delete following completion of #4001
    pathlib.Path(doc_path, "memory", "chat_message.ipynb").resolve(), # todo: hf model repo is gated, pick a new model
    pathlib.Path(doc_path, "orchestrators", "HITL_Scoring_Orchestrator.ipynb").resolve(), # requires human input
    pathlib.Path(doc_path, "targets", "4_non_llm_targets.ipynb").resolve(),
    pathlib.Path(doc_path, "targets", "playwright_target.ipynb").resolve(),
]

# Convert to string for comparison
skipped_files = [str(f) for f in skipped_files]

# Extract full path for each notebook within doc_path
nb_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(doc_path)) for f in fn if f.endswith(".ipynb")]

@pytest.mark.parametrize(
    "file_path",
    [f for f in nb_files if f not in skipped_files],
)
def test_execute_notebooks(file_path):
    nb_path = pathlib.Path(file_path).resolve()
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600)

    # Execute notebook, test will throw exception if any cell fails
    ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
