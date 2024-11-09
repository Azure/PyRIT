# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

from appdirs import user_data_dir


def get_default_results_path() -> pathlib.Path:
    if in_git_repo():
        return pathlib.Path(PYRIT_PATH, "..", "results").resolve()
    else:
        return pathlib.Path(user_data_dir("results", "pyrit")).resolve()


def in_git_repo() -> bool:
    return pathlib.Path(HOME_PATH, ".git").exists()


PYRIT_PATH = pathlib.Path(__file__, "..", "..").resolve()

DATASETS_PATH = pathlib.Path(PYRIT_PATH, "datasets").resolve()
CONTENT_CLASSIFIERS_PATH = pathlib.Path(DATASETS_PATH, "score", "content_classifiers").resolve()
LIKERT_SCALES_PATH = pathlib.Path(DATASETS_PATH, "score", "likert_scales").resolve()
SCALES_PATH = pathlib.Path(DATASETS_PATH, "score", "scales").resolve()

RED_TEAM_ORCHESTRATOR_PATH = pathlib.Path(DATASETS_PATH, "orchestrators", "red_teaming").resolve()

# Points to the root of the project
HOME_PATH = pathlib.Path(PYRIT_PATH, "..").resolve()

# Path to where all the results files and database will be stored
RESULTS_PATH = get_default_results_path()
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Path to where the logs are located
LOG_PATH = pathlib.Path(RESULTS_PATH, "logs.txt").resolve()
LOG_PATH.touch(exist_ok=True)
