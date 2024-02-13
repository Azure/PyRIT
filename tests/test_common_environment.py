# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.common import environment_variables
from pyrit.common.net_utility import make_request_and_raise_if_error


def test_environment_variables_prefers_passed():
    os.environ["TEST_ENV_VAR"] = "fail"
    assert environment_variables.get_required_value("TEST_ENV_VAR", "passed") == "passed"


def test_environment_variables_uses_default():
    os.environ["TEST_ENV_VAR"] = "default"
    assert environment_variables.get_required_value("TEST_ENV_VAR", "") == "default"


def test_environment_variables_throws_if_not_set():
    os.environ["TEST_ENV_VAR"] = ""
    with pytest.raises(ValueError):
        environment_variables.get_required_value("TEST_ENV_VAR", "") == "default"

