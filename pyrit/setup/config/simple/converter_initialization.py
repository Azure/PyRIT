# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This is a good default converter configuration for PyRIT.
"""
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.pyrit_default_value import set_default_value

default_converter_target = OpenAIChatTarget(
    temperature=0.7,
)


set_default_value(class_type=PromptConverter, parameter_name="converter_target", value=default_converter_target)
