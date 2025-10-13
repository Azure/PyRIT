# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures harm bench for testing (only the first eight)

"""

from pyrit.datasets import fetch_harmbench_dataset

_objectives = fetch_harmbench_dataset().get_values(first=8)


# Define the dataset configuration
# This dictionary is used by DataSetFactory to create the attack instance
dataset_config = {
    "objectives": _objectives,
}
