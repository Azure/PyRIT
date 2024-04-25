# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

PYRIT_PATH = pathlib.Path(__file__, "..", "..").resolve()
# Path to where all the results files will be stores
RESULTS_PATH = pathlib.Path(PYRIT_PATH, "..", "results").resolve()
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
# Path to where the logs are located
LOG_PATH = pathlib.Path(PYRIT_PATH, "..", "results", "logs.txt").resolve()
LOG_PATH.touch(exist_ok=True)
# Path to where docs exist
DOCS_PATH = pathlib.Path(PYRIT_PATH, "..", "doc").resolve()
DOCS_PATH.mkdir(parents=True, exist_ok=True)

DATASETS_PATH = pathlib.Path(PYRIT_PATH, "datasets").resolve()
CONTENT_CLASSIFIERS_PATH = pathlib.Path(DATASETS_PATH, "score", "content_classifiers").resolve()
LIKERT_SCALES_PATH = pathlib.Path(DATASETS_PATH, "score", "likert_scales").resolve()
SCORING_INSTRUCTIONS_PATH = pathlib.Path(DATASETS_PATH, "score", "scoring_instructions").resolve()

# Points to the root of the project
HOME_PATH = pathlib.Path(PYRIT_PATH, "..").resolve()
