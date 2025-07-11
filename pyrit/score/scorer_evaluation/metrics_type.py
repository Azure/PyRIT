# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class MetricsType(Enum):
    """
    Enum representing the type of metrics when evaluating scorers on human-labeled datasets.
    """

    HARM = "harm"
    OBJECTIVE = "objective"
