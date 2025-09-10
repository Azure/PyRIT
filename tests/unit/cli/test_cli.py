# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import re
import shlex
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pyrit.cli.__main__ import main
from pyrit.executor.attack import (
    CrescendoAttack,
    PromptSendingAttack,
    RedTeamingAttack,
    TreeOfAttacksWithPruningAttack,
)

test_cases_success = [
    (
        "--config-file 'tests/unit/cli/prompt_send_success.yaml'",
        [PromptSendingAttack],
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_success_converters_default.yaml'",
        [PromptSendingAttack],
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_success_converters_custom_target.yaml'",
        [PromptSendingAttack],
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_success_converters_llm_mixed_target.yaml'",
        [PromptSendingAttack],
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_success_converters_no_target.yaml'",
        [PromptSendingAttack],
    ),
    ("--config-file 'tests/unit/cli/multi_turn_rto_success.yaml'", [RedTeamingAttack]),
    ("--config-file 'tests/unit/cli/multi_turn_rto_args_success.yaml'", [RedTeamingAttack]),
    ("--config-file 'tests/unit/cli/multi_turn_crescendo_success.yaml'", [CrescendoAttack]),
    (
        "--config-file 'tests/unit/cli/multi_turn_crescendo_args_success.yaml'",
        [CrescendoAttack],
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_tap_success.yaml'",
        [TreeOfAttacksWithPruningAttack],
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_tap_args_success.yaml'",
        [TreeOfAttacksWithPruningAttack],
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_multiple_attacks_args_success.yaml'",
        [TreeOfAttacksWithPruningAttack, CrescendoAttack, RedTeamingAttack],
    ),
    (
        "--config-file 'tests/unit/cli/mixed_multiple_attacks_args_success.yaml'",
        [
            PromptSendingAttack,
            TreeOfAttacksWithPruningAttack,
            CrescendoAttack,
            RedTeamingAttack,
        ],
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_target_and_scorer_args_success.yaml'",
        [
            CrescendoAttack,
        ],
    ),
]


test_cases_sys_exit = [
    (
        "",  # No argument passed
        "the following arguments are required: --config-file",
    ),
    (
        "-config-file './some/path/to/a/config.yml'",  # Wrong flag passed
        "the following arguments are required: --config-file",
    ),
]

test_cases_error = [
    (
        "--config-file 'tests/unit/cli/prompt_send_no_objective_target.yaml'",
        "objective_target\n  Field required",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_objective_target_type.yaml'",
        "objective_target\n  Value error, Field 'objective_target' must be a dictionary.\n"
        "Example:\n  objective_target:\n    type: OpenAIChatTarget",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenarios.yaml'",
        "scenarios\n  Input should be a valid list",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenarios_key.yaml'",
        "scenarios\n  Field required",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenario_type.yaml'",
        "scenarios.0.type\n  Field required",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scoring_target.yaml'",
        "Scorer requires a scoring_target to be defined. "
        "Alternatively, the adversarial_target can be used for scoring purposes, but none was provided.",
        KeyError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_converter_target.yaml'",
        "Converter requires a converter_target to be defined. "
        "Alternatively, the adversarial_target can be used for scoring purposes, but none was provided.",
        KeyError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_converters_wrong_arg.yaml'",
        "TranslationConverter.__init__() got an unexpected keyword argument 'wrong_arg'",
        TypeError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_converters_missing_arg.yaml'",
        "TranslationConverter.__init__() missing 1 required keyword-only argument: 'language'",
        TypeError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_rto_wrong_arg.yaml'",
        "Failed to instantiate scenario 'RedTeamingAttack': RedTeamingAttack.__init__() "
        "got an unexpected keyword argument 'wrong_arg'",
        ValueError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_crescendo_wrong_arg.yaml'",
        "Failed to instantiate scenario 'CrescendoAttack': CrescendoAttack.__init__() "
        "got an unexpected keyword argument 'wrong_arg'",
        ValueError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_tap_wrong_arg.yaml'",
        "Failed to instantiate scenario 'TreeOfAttacksWithPruningAttack': "
        "TreeOfAttacksWithPruningAttack.__init__() "
        "got an unexpected keyword argument 'wrong_arg'",
        ValueError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_success_invalid_exec.yaml'",
        "execution_settings.type\n  Input should be 'local' ",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_bad_db_type.yaml'",
        "database.type\n  Input should be 'InMemory', 'SQLite' or 'AzureSQL' ",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_db_with_no_type.yaml'",
        "database.type\n  Field required",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_db.yaml'",
        "database\n  Field required",
        ValidationError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_scoring_target_wrong_arg.yaml'",
        "OpenAITarget.__init__() got an unexpected keyword argument 'nonsense_arg'",
        TypeError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_objective_scorer_wrong_arg.yaml'",
        "SelfAskTrueFalseScorer.__init__() got an unexpected keyword argument 'nonsense_arg'",
        TypeError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_adversarial_target_wrong_arg.yaml'",
        "OpenAITarget.__init__() got an unexpected keyword argument 'nonsense_arg'",
        TypeError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_objective_target_wrong_arg.yaml'",
        "OpenAITarget.__init__() got an unexpected keyword argument 'nonsense_arg'",
        TypeError,
    ),
]


@pytest.mark.parametrize("command, attack_classes", test_cases_success)
@patch("pyrit.common.default_values.get_required_value", return_value="value")
def test_cli_success(get_required_value, command, attack_classes):
    # Patching the request sending functionality since we don't want to test the attack,
    # but just the CLI part.

    with contextlib.ExitStack() as stack:
        for attack_class in attack_classes:
            stack.enter_context(patch.object(attack_class.__base__.__base__, "execute_async"))
        main(shlex.split(command))


@pytest.mark.parametrize("command, expected_output", test_cases_sys_exit)
def test_cli_sys_exit(capsys, command, expected_output):
    with pytest.raises(SystemExit):  # Expecting SystemExit due to argparse error
        main(shlex.split(command))
    captured = capsys.readouterr()  # Capture both stdout and stderr
    output = captured.out + captured.err  # Combine stdout and stderr
    assert expected_output in output


@pytest.mark.parametrize("command, expected_output, error_type", test_cases_error)
@patch("pyrit.common.default_values.get_required_value", return_value="value")
def test_cli_error(get_required_value, command, expected_output, error_type):
    with pytest.raises(error_type, match=re.escape(expected_output)):
        main(shlex.split(command))
