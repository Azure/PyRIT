# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import pytest

from pyrit.common.apply_defaults import reset_default_values
from pyrit.setup.initializers import SimpleInitializer


class TestSimpleInitializer:
    """Tests for SimpleInitializer class - basic functionality."""

    def test_simple_initializer_can_be_created(self):
        """Test that SimpleInitializer can be instantiated."""
        init = SimpleInitializer()
        assert init is not None
        assert init.name == "Simple Complete Configuration"
        assert init.execution_order == 1


@pytest.mark.usefixtures("patch_central_database")
class TestSimpleInitializerInitialize:
    """Tests for SimpleInitializer.initialize method."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Set up required env vars for OpenAI
        os.environ["OPENAI_CHAT_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["OPENAI_CHAT_KEY"] = "test_key"
        # Clean up globals
        for attr in ["default_converter_target", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        # Clean up env vars
        for var in ["OPENAI_CHAT_ENDPOINT", "OPENAI_CHAT_KEY"]:
            if var in os.environ:
                del os.environ[var]
        # Clean up globals
        for attr in ["default_converter_target", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def test_initialize_runs_without_error(self):
        """Test that initialize runs without errors."""
        init = SimpleInitializer()
        # Should not raise any errors
        init.initialize()

    def test_get_info_after_initialize_has_populated_data(self):
        """Test that get_info() returns populated data after initialization."""
        init = SimpleInitializer()
        init.initialize()

        info = SimpleInitializer.get_info()

        # Verify basic structure
        assert isinstance(info, dict)
        assert "name" in info
        assert "default_values" in info
        assert "global_variables" in info

        # Verify default_values list is populated and not empty
        assert isinstance(info["default_values"], list)
        assert len(info["default_values"]) > 0, "default_values should be populated after initialization"

        # Verify expected default values are present
        default_values_str = str(info["default_values"])
        assert "PromptConverter.converter_target" in default_values_str
        assert "PromptSendingAttack.attack_scoring_config" in default_values_str

        # Verify global_variables list is populated and not empty
        assert isinstance(info["global_variables"], list)
        assert len(info["global_variables"]) > 0, "global_variables should be populated after initialization"

        # Verify expected global variables are present
        assert "default_converter_target" in info["global_variables"]
        assert "default_objective_scorer" in info["global_variables"]
        assert "adversarial_config" in info["global_variables"]


class TestSimpleInitializerGetInfo:
    """Tests for SimpleInitializer.get_info method - basic functionality."""

    def test_get_info_returns_expected_structure(self):
        """Test that get_info returns expected structure."""
        info = SimpleInitializer.get_info()

        assert isinstance(info, dict)
        assert info["name"] == "Simple Complete Configuration"
        assert info["class"] == "SimpleInitializer"
        assert "required_env_vars" in info
        assert "OPENAI_CHAT_ENDPOINT" in info["required_env_vars"]
