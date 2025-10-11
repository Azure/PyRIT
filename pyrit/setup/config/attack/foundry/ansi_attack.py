# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures an AnsiAttack.

It can be modified to set up convenient variables, default values, or helper functions.
"""

from pyrit.executor.attack import PromptSendingAttack, AttackConverterConfig, AttackFactory
from pyrit.prompt_converter import AnsiAttackConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.setup import set_default_value

_converters = PromptConverterConfiguration.from_converters(converters=[AnsiAttackConverter()])
_attack_converter_config = AttackConverterConfig(request_converters=_converters)


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