# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This is a good default scorer configuration for PyRIT. It sets up objective targets and
adversarial targets for Executors.
"""
import os
from pyrit.prompt_target import OpenAIChatTarget

default_converter_target = OpenAIChatTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
    temperature=0.5
)