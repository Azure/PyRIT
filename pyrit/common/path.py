# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

from appdirs import user_data_dir


def get_default_data_path(dir: str) -> pathlib.Path:
    if in_git_repo():
        return pathlib.Path(PYRIT_PATH, "..", dir).resolve()
    else:
        return pathlib.Path(user_data_dir(dir, "pyrit")).resolve()


def in_git_repo() -> bool:
    return pathlib.Path(HOME_PATH, ".git").exists()


PYRIT_PATH = pathlib.Path(__file__, "..", "..").resolve()

DOCS_PATH = pathlib.Path(PYRIT_PATH, "..", "doc").resolve()

DOCS_CODE_PATH = pathlib.Path(PYRIT_PATH, "..", "doc", "code").resolve()
DATASETS_PATH = pathlib.Path(PYRIT_PATH, "datasets").resolve()
CONTENT_CLASSIFIERS_PATH = pathlib.Path(DATASETS_PATH, "score", "content_classifiers").resolve()
LIKERT_SCALES_PATH = pathlib.Path(DATASETS_PATH, "score", "likert_scales").resolve()
SCALES_PATH = pathlib.Path(DATASETS_PATH, "score", "scales").resolve()
SCORER_EVALS_PATH = pathlib.Path(DATASETS_PATH, "score", "scorer_evals").resolve()
SCORER_EVALS_HARM_PATH = pathlib.Path(SCORER_EVALS_PATH, "harm").resolve()
SCORER_EVALS_OBJECTIVE_PATH = pathlib.Path(SCORER_EVALS_PATH, "objective").resolve()

RED_TEAM_ORCHESTRATOR_PATH = pathlib.Path(DATASETS_PATH, "orchestrators", "red_teaming").resolve()

# Points to the root of the project
HOME_PATH = pathlib.Path(PYRIT_PATH, "..").resolve()

# Path to where all the seed prompt entry and prompt memory entry files and database file will be stored
DB_DATA_PATH = get_default_data_path("dbdata")
DB_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Path to where the logs are located
LOG_PATH = pathlib.Path(DB_DATA_PATH, "logs.txt").resolve()
LOG_PATH.touch(exist_ok=True)

# Dictionary of default PyRIT paths used primarily for rendering jinja templates
PATHS_DICT = {
    "content_classifiers_path": CONTENT_CLASSIFIERS_PATH,
    "datasets_path": DATASETS_PATH,
    "db_data_path": DB_DATA_PATH,
    "docs_code_path": DOCS_CODE_PATH,
    "docs_path": DOCS_PATH,
    "likert_scales_path": LIKERT_SCALES_PATH,
    "log_path": LOG_PATH,
    "pyrit_home_path": HOME_PATH,
    "pyrit_path": PYRIT_PATH,
    "red_team_orchestrator_path": RED_TEAM_ORCHESTRATOR_PATH,
    "scales_path": SCALES_PATH,
    "scorer_evals_path": SCORER_EVALS_PATH,
    "scorer_evals_harm_path": SCORER_EVALS_HARM_PATH,
    "scorer_evals_objective_path": SCORER_EVALS_OBJECTIVE_PATH,
}
