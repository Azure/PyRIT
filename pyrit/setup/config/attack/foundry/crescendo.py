# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures a Crescendo attack.

This configuration file defines the parameters for a CrescendoAttack
with custom max_turns and max_backtracks settings. It can be modified
to adjust these parameters or add additional configurations.
"""

# Define the attack configuration
# This dictionary is used by AttackFactory to create the attack instance
attack_config = {
    "attack_type": "CrescendoAttack",
    "max_turns": 3,
    "max_backtracks": 2,
}