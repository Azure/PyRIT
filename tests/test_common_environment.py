# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.common import default_values


def test_environment_variables_prefers_passed():
    os.environ["TEST_ENV_VAR"] = "fail"
    assert default_values.get_required_value(env_var_name="TEST_ENV_VAR", passed_value="passed") == "passed"


def test_environment_variables_uses_default():
    os.environ["TEST_ENV_VAR"] = "default"
    assert default_values.get_required_value(env_var_name="TEST_ENV_VAR", passed_value="") == "default"


def test_environment_variables_throws_if_not_set():
    os.environ["TEST_ENV_VAR"] = ""
    with pytest.raises(ValueError):
        default_values.get_required_value(env_var_name="TEST_ENV_VAR", passed_value="") == "default"
