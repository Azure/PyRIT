# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures a Tense attack.

This configuration file defines the parameters for a PromptSendingAttack
that uses tense conversion (randomly selected between past and future).
It can be modified to adjust the tense or converter settings.

Note: This configuration requires converter_target defaults to be set via
initialize_pyrit() before use.
"""
import random
from typing import List, cast

from pyrit.executor.attack import AttackConverterConfig
from pyrit.prompt_converter import PromptConverter, TenseConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration

# Randomly select tense
_tense = random.choice(["past", "future"])

# Create the converter configuration
# Note: TenseConverter requires converter_target, which should be set via defaults
_converter_configurations = PromptConverterConfiguration.from_converters(
    converters=cast(List[PromptConverter], [TenseConverter(tense=_tense, converter_target=None)])  # Uses default
)

_attack_converter_config = AttackConverterConfig(request_converters=_converter_configurations)

# Define the attack configuration
# This dictionary is used by AttackFactory to create the attack instance
attack_config = {
    "attack_type": "PromptSendingAttack",
    "attack_converter_config": _attack_converter_config,
}
