# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This is a good default scorer configuration for PyRIT. It sets up objective targets and
adversarial targets for Executors.
"""
import os

from pyrit.executor.attack import AttackAdversarialConfig, MultiPromptSendingAttack
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.prompt_target import OpenAIChatTarget

from pyrit.setup import set_default_value


_adversarial_config = AttackAdversarialConfig(
    target=OpenAIChatTarget(
        endpoint=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
        temperature=0.5,
    )        
)

set_default_value(
    class_type=CrescendoAttack, parameter_name="attack_adversarial_config", value=_adversarial_config
)

