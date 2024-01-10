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

DATASETS_PATH = pathlib.Path(PYRIT_PATH, "..", "datasets").resolve()

# Points to the root of the project
HOME_PATH = pathlib.Path(PYRIT_PATH, "..").resolve()
