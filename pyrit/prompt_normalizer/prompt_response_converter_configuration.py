# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter


@dataclass
class PromptResponseConverterConfiguration:
    """
    Represents the configuration for a prompt response converter.

    The list of converters are applied to a response, which can have multiple response pieces.
    indexes_to_apply are which pieces to apply to. By default, all indexes are applied.
    prompt_data_types_to_apply are the types of the responses to apply the converters.
    """

    converters: list[PromptConverter]
    indexes_to_apply: list[int] = None
    prompt_data_types_to_apply: list[PromptDataType] = None
