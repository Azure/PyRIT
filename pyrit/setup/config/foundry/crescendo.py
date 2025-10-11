# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures a Crescendo attack.

It can be modified to set up convenient variables, default values, or helper functions.
"""

from pyrit.executor.attack import CrescendoAttack, AttackFactory
from pyrit.setup import set_default_value


# Configure default values for CrescendoAttack (and subclasses)
set_default_value(
    class_type=CrescendoAttack,
    include_subclasses=False,
    parameter_name="max_turns",
    value=3,
)

# Configure default values for CrescendoAttack (and subclasses)
set_default_value(
    class_type=CrescendoAttack,
    include_subclasses=False,
    parameter_name="max_backtracks",
    value=2,
)

set_default_value(
    class_type=AttackFactory,
    include_subclasses=False,
    parameter_name="attack_type",
    value="CrescendoAttack",
)