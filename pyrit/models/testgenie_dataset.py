# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict


class TestGenieDataset(BaseModel):
    """
    Represents a dataset for question answering.

    Attributes:
        name (str): The name of the dataset.
        version (str): The version of the dataset.
        description (str): A description of the dataset.
        author (str): The author of the dataset.
        group (str): The group associated with the dataset.
        source (str): The source of the dataset.
        questions (list[QuestionAnsweringEntry]): A list of question models.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = ""
    version: str = ""
    description: str = ""
    author: str = ""
    group: str = ""
    source: str = ""
