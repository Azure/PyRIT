# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import re
import shlex
from unittest.mock import patch

import pytest

from pyrit.cli.__main__ import main
from pyrit.orchestrator import PromptSendingOrchestrator, CrescendoOrchestrator, RedTeamingOrchestrator, TreeOfAttacksWithPruningOrchestrator

test_cases_success = [
    (
        "--config-file 'tests/unit/cli/prompt_send_success.yaml'",
        [PromptSendingOrchestrator],
        ["send_normalizer_requests_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_rto_success.yaml'",
        [RedTeamingOrchestrator],
        ["run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_rto_args_success.yaml'",
        [RedTeamingOrchestrator],
        ["run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_crescendo_success.yaml'",
        [CrescendoOrchestrator],
        ["run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_crescendo_args_success.yaml'",
        [CrescendoOrchestrator],
        ["run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_tap_success.yaml'",
        [TreeOfAttacksWithPruningOrchestrator],
        ["run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_tap_args_success.yaml'",
        [TreeOfAttacksWithPruningOrchestrator],
        ["run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_multiple_orchestrators_args_success.yaml'",
        [TreeOfAttacksWithPruningOrchestrator, CrescendoOrchestrator, RedTeamingOrchestrator],
        ["run_attack_async", "run_attack_async", "run_attack_async"]
    ),
    (
        "--config-file 'tests/unit/cli/mixed_multiple_orchestrators_args_success.yaml'",
        [PromptSendingOrchestrator, TreeOfAttacksWithPruningOrchestrator, CrescendoOrchestrator, RedTeamingOrchestrator],
        ["send_normalizer_requests_async", "run_attack_async", "run_attack_async", "run_attack_async"]
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
        "Configuration file must contain a 'objective_target' key.",
        KeyError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_objective_target_type.yaml'",
        "Target objective_target must contain a 'type' key.",
        KeyError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenarios.yaml'",
        "Scenarios list is empty.",
        ValueError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenarios_key.yaml'",
        "Configuration file must contain a 'scenarios' key.",
        KeyError,
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenario_type.yaml'",
        "Scenario must contain a 'type' key.",
        KeyError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_rto_wrong_arg.yaml'",
        "Failed to validate scenario RedTeamingOrchestrator: RedTeamingOrchestrator.__init__() got an unexpected keyword argument 'wrong_arg'",
        ValueError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_crescendo_wrong_arg.yaml'",
        "Failed to validate scenario CrescendoOrchestrator: CrescendoOrchestrator.__init__() got an unexpected keyword argument 'wrong_arg'",
        ValueError,
    ),
    (
        "--config-file 'tests/unit/cli/multi_turn_tap_wrong_arg.yaml'",
        "Failed to validate scenario TreeOfAttacksWithPruningOrchestrator: TreeOfAttacksWithPruningOrchestrator.__init__() got an unexpected keyword argument 'wrong_arg'",
        ValueError,
    ),
]


@pytest.mark.parametrize("command, orchestrator_classes, methods", test_cases_success)
# Patching OpenAI target initialization which depends on environment variables
# which we are not providing here.
@patch("pyrit.prompt_target.OpenAIChatTarget._initialize_azure_vars")
def test_cli_pso_success(init_method, command, orchestrator_classes, methods):
    # Patching the request sending functionality since we don't want to test the orchestrator,
    # but just the CLI part.
    with contextlib.ExitStack() as stack:
        for orchestrator_class, method in zip(orchestrator_classes, methods):
            stack.enter_context(patch.object(orchestrator_class, method))
        main(shlex.split(command))


@pytest.mark.parametrize("command, expected_output", test_cases_sys_exit)
def test_cli_sys_exit(capsys, command, expected_output):
    with pytest.raises(SystemExit):  # Expecting SystemExit due to argparse error
        main(shlex.split(command))
    captured = capsys.readouterr()  # Capture both stdout and stderr
    output = captured.out + captured.err  # Combine stdout and stderr
    assert expected_output in output


@pytest.mark.parametrize("command, expected_output, error_type", test_cases_error)
# Patching OpenAI target initialization which depends on environment variables
# which we are not providing here.
@patch("pyrit.prompt_target.OpenAIChatTarget._initialize_azure_vars")
def test_cli_error(init_method, command, expected_output, error_type):
    with pytest.raises(error_type, match=re.escape(expected_output)):
        main(shlex.split(command))
