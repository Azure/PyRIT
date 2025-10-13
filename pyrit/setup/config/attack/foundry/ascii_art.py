# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures an AsciiArt attack.

This configuration file defines the parameters for a PromptSendingAttack
that uses ASCII art conversion. It can be modified to adjust the converter
settings or add additional configurations.
"""

from pyrit.executor.attack import AttackConverterConfig
from pyrit.prompt_converter import AsciiArtConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration

# Create the converter configuration
_converters = PromptConverterConfiguration.from_converters(converters=[AsciiArtConverter()])
_attack_converter_config = AttackConverterConfig(request_converters=_converters)

# Define the attack configuration
# This dictionary is used by AttackFactory to create the attack instance
attack_config = {
    "attack_type": "PromptSendingAttack",
    "attack_converter_config": _attack_converter_config,
}
