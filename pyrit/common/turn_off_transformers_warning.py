# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

# Most people install PyRIT without torch so transformers prints a warning
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "True"
