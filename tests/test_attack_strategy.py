# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common.path import DATASETS_PATH
from pyrit.models import AttackStrategy


def test_attack_strategy_strings():
    assert "my strategy my objective" == str(
        AttackStrategy(strategy="my strategy {{ conversation_objective }}", conversation_objective="my objective")
    )


def test_attack_strategy_from_file():
    strategy_path = DATASETS_PATH / "orchestrators" / "red_teaming" / "text_generation.yaml"
    with open(strategy_path, "r") as strategy_file:
        strategy = strategy_file.read()
        string_before_template = "value: |\n  "
        strategy_template = strategy[strategy.find(string_before_template) + len(string_before_template) :]
        strategy_template = strategy_template.replace("\n  ", "\n")
        strategy_template = strategy_template.rstrip()
    assert strategy_template.replace("{{ conversation_objective }}", "my objective") == str(
        AttackStrategy(strategy=strategy_path, conversation_objective="my objective")
    )
