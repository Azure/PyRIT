# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt


def test_system_prompt_from_file():
    strategy_path = DATASETS_PATH / "orchestrators" / "red_teaming" / "text_generation.yaml"
    with open(strategy_path, "r") as strategy_file:
        strategy = strategy_file.read()
        string_before_template = "value: |\n  "
        strategy_template = strategy[strategy.find(string_before_template) + len(string_before_template) :]
        strategy_template = strategy_template.replace("\n  ", "\n")
        strategy_template = strategy_template.rstrip()
        seed_prompt = SeedPrompt.from_yaml_file(strategy_path)
    assert strategy_template.replace("{{ objective }}", "my objective") == str(
        seed_prompt.render_template_value(objective="my objective")
    )
