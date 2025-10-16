# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This is a good default converter configuration for PyRIT.
"""
import os

from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.pyrit_default_value import set_default_value

default_converter_target = OpenAIChatTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
    temperature=0.5,
)


set_default_value(class_type=PromptConverter, parameter_name="converter_target", value=default_converter_target)
