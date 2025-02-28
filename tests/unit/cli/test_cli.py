# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shlex
from unittest.mock import patch

import pytest

from pyrit.cli.__main__ import main
from pyrit.orchestrator import PromptSendingOrchestrator

test_cases_success = ["--config-file 'tests/unit/cli/prompt_send_success.yaml'"]


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
]


@pytest.mark.parametrize("command", test_cases_success)
def test_cli_success(command):
    # Patching the request sending functionality since we don't want to test the orchestrator,
    # but just the CLI part.
    # And patching OpenAI target initialization which depends on environment variables
    # which we are not providing here.
    with (
        patch.object(PromptSendingOrchestrator, "send_normalizer_requests_async"),
        patch("pyrit.common.default_values.get_required_value", return_value="value"),
    ):
        main(shlex.split(command))


@pytest.mark.parametrize("command, expected_output", test_cases_sys_exit)
def test_cli_sys_exit(capsys, command, expected_output):
    with pytest.raises(SystemExit):  # Expecting SystemExit due to argparse error
        main(shlex.split(command))
    captured = capsys.readouterr()  # Capture both stdout and stderr
    output = captured.out + captured.err  # Combine stdout and stderr
    assert expected_output in output


@pytest.mark.parametrize("command, expected_output, error_type", test_cases_error)
def test_cli_error(command, expected_output, error_type):
    # Patching OpenAI target initialization which depends on environment variables
    # which we are not providing here.
    with patch("pyrit.common.default_values.get_required_value", return_value="value"):
        with pytest.raises(error_type, match=expected_output):
            main(shlex.split(command))
