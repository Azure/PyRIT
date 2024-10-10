# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPromptTemplate
from pyrit.prompt_converter.fuzzer_converter.fuzzer_converter_base import FuzzerConverter
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget


class FuzzerSimilarConverter(FuzzerConverter):
    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: SeedPromptTemplate = None):
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "fuzzer_converters" / "similar_converter.yaml"
            )
        )
        super().__init__(converter_target=converter_target, prompt_template=prompt_template)
