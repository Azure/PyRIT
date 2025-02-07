# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import shlex

from pyrit.cli.__main__ import main


test_cases = [
]


test_cases_sys_exit = [
    (
        "",  # No argument passed
        "the following arguments are required: --config-file",
        Sys
    ),
    (
        "-config-file './some/path/to/a/config.yml'",  # Wrong flag passed 
        "the following arguments are required: --config-file",
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_objective_target.yaml'", 
        "Configuration file must contain an 'objective_target' key.",
    ),
    (
        "--config-file 'tests/unit/cli/prompt_send_no_scenarios.yaml'", 
        "Configuration file must contain a 'scenarios' key.",
    ),
]


@pytest.mark.parametrize("command, expected_output", test_cases)
def test_cli(capsys, command, expected_output):
    main(shlex.split(command))
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert expected_output in output


@pytest.mark.parametrize("command, expected_output,", test_cases_sys_exit)
def test_cli_sys_exit(capsys, command, expected_output):
    with pytest.raises(BaseException):  # Expecting SystemExit due to argparse error
        main(shlex.split(command))
    captured = capsys.readouterr()  # Capture both stdout and stderr
    output = captured.out + captured.err  # Combine stdout and stderr
    assert expected_output in output