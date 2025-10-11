# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures an AsciiArt.

It can be modified to set up convenient variables, default values, or helper functions.
"""
import random
from typing import cast, List

from pyrit.executor.attack import PromptSendingAttack, AttackConverterConfig, AttackFactory
from pyrit.prompt_converter import PromptConverter, TenseConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.setup import set_default_value


_tense = random.choice(["past", "future"])

_converter_configurations = PromptConverterConfiguration.from_converters(
    converters=cast(List[PromptConverter], [
        TenseConverter(
            tense=_tense,
            converter_target=None  # Uses default from converter_initialization.py
        )
    ])
)

_attack_converter_config = AttackConverterConfig(
    request_converters=_converter_configurations
)

# Configure default values for PromptSendingAttack (and subclasses)
set_default_value(
    class_type=PromptSendingAttack,
    include_subclasses=False,
    parameter_name="attack_converter_config",
    value=_attack_converter_config,
)

set_default_value(
    class_type=AttackFactory,
    include_subclasses=False,
    parameter_name="attack_type",
    value="PromptSendingAttack",
)