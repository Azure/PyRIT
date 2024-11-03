# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SystemPromptWithObjective


def test_system_prompt_from_file():
    strategy_path = DATASETS_PATH / "orchestrators" / "red_teaming" / "text_generation.yaml"
    with open(strategy_path, "r") as strategy_file:
        strategy = strategy_file.read()
        string_before_template = "value: |\n  "
        strategy_template = strategy[strategy.find(string_before_template) + len(string_before_template) :]
        strategy_template = strategy_template.replace("\n  ", "\n")
<<<<<<< HEAD:tests/test_prompt_template.py
    assert strategy_template.replace("{{ objective }}", "my objective") == str(
        SystemPromptWithObjective(path=strategy_path, objective="my objective")
=======
        strategy_template = strategy_template.rstrip()
    assert strategy_template.replace("{{ conversation_objective }}", "my objective") == str(
        AttackStrategy(strategy=strategy_path, conversation_objective="my objective")
>>>>>>> main:tests/test_attack_strategy.py
    )
