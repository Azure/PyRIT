# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This is a good default scorer configuration for PyRIT. It sets up objective targets and
adversarial targets for Executors.
"""

from pyrit.executor.attack import AttackAdversarialConfig
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import set_default_value, set_global_variable

adversarial_config = AttackAdversarialConfig(
    target=OpenAIChatTarget(
        temperature=0.5,
    )
)

set_global_variable(name="adversarial_config", value=adversarial_config)
set_default_value(class_type=CrescendoAttack, parameter_name="attack_adversarial_config", value=adversarial_config)
