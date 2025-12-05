# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import pytest

from pyrit.common.apply_defaults import reset_default_values
from pyrit.setup.initializers import AIRTInitializer


class TestAIRTInitializer:
    """Tests for AIRTInitializer class - basic functionality."""

    def test_airt_initializer_can_be_created(self):
        """Test that AIRTInitializer can be instantiated."""
        init = AIRTInitializer()
        assert init is not None
        assert init.name == "AIRT Default Configuration"
        assert init.execution_order == 1

    def test_airt_initializer_description(self):
        """Test that AIRTInitializer has the correct description."""
        init = AIRTInitializer()
        assert "AI Red Team" in init.description
        assert "Azure OpenAI" in init.description


@pytest.mark.usefixtures("patch_central_database")
class TestAIRTInitializerInitialize:
    """Tests for AIRTInitializer.initialize method."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Set up required env vars for AIRT
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"] = "https://test-converter.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_converter_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test-scorer.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_scorer_key"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test-safety.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "test_safety_key"
        # Clean up globals
        for attr in [
            "default_converter_target",
            "default_harm_scorer",
            "default_objective_scorer",
            "adversarial_config",
        ]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        # Clean up env vars
        for var in [
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2",
            "AZURE_CONTENT_SAFETY_API_ENDPOINT",
            "AZURE_CONTENT_SAFETY_API_KEY",
        ]:
            if var in os.environ:
                del os.environ[var]
        # Clean up globals
        for attr in [
            "default_converter_target",
            "default_harm_scorer",
            "default_objective_scorer",
            "adversarial_config",
        ]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def test_initialize_runs_without_error(self):
        """Test that initialize runs without errors."""
        init = AIRTInitializer()
        # Should not raise any errors
        init.initialize()

    def test_get_info_after_initialize_has_populated_data(self):
        """Test that get_info() returns populated data after initialization."""
        init = AIRTInitializer()
        init.initialize()

        info = AIRTInitializer.get_info()

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
        assert "PromptSendingAttack.attack_adversarial_config" in default_values_str

        # Verify global_variables list is populated and not empty
        assert isinstance(info["global_variables"], list)
        assert len(info["global_variables"]) > 0, "global_variables should be populated after initialization"

        # Verify expected global variables are present
        assert "default_converter_target" in info["global_variables"]
        assert "default_harm_scorer" in info["global_variables"]
        assert "default_objective_scorer" in info["global_variables"]
        assert "adversarial_config" in info["global_variables"]

    def test_validate_missing_env_vars_raises_error(self):
        """Test that validate raises error when required env vars are missing."""
        # Remove one required env var
        del os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"]

        init = AIRTInitializer()
        with pytest.raises(ValueError) as exc_info:
            init.validate()

        error_message = str(exc_info.value)
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT" in error_message
        assert "environment variables" in error_message

    def test_validate_missing_multiple_env_vars_raises_error(self):
        """Test that validate raises error listing all missing env vars."""
        # Remove multiple required env vars
        del os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"]
        del os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"]

        init = AIRTInitializer()
        with pytest.raises(ValueError) as exc_info:
            init.validate()

        error_message = str(exc_info.value)
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT" in error_message
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY" in error_message


class TestAIRTInitializerGetInfo:
    """Tests for AIRTInitializer.get_info method - basic functionality."""

    def test_get_info_returns_expected_structure(self):
        """Test that get_info returns expected structure."""
        info = AIRTInitializer.get_info()

        assert isinstance(info, dict)
        assert info["name"] == "AIRT Default Configuration"
        assert info["class"] == "AIRTInitializer"
        assert "required_env_vars" in info
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT" in info["required_env_vars"]
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY" in info["required_env_vars"]
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2" in info["required_env_vars"]
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2" in info["required_env_vars"]
        assert "AZURE_CONTENT_SAFETY_API_ENDPOINT" in info["required_env_vars"]
        assert "AZURE_CONTENT_SAFETY_API_KEY" in info["required_env_vars"]

    def test_get_info_includes_description(self):
        """Test that get_info includes the description field."""
        info = AIRTInitializer.get_info()

        assert "description" in info
        assert isinstance(info["description"], str)
        assert len(info["description"]) > 0
