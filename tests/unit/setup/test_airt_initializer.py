# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from unittest import mock

import pytest

from pyrit.common.apply_defaults import get_global_default_values, reset_default_values
from pyrit.setup.initializers.airt import AIRTInitializer


class TestAIRTInitializer:
    """Tests for AIRTInitializer class."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Clean up globals
        for attr in ["default_converter_target", "default_harm_scorer", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        for attr in ["default_converter_target", "default_harm_scorer", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def test_airt_initializer_can_be_created(self):
        """Test that AIRTInitializer can be instantiated."""
        init = AIRTInitializer()
        assert init is not None

    def test_name_property(self):
        """Test the name property."""
        init = AIRTInitializer()
        assert init.name == "AIRT Default Configuration"

    def test_description_property(self):
        """Test the description property."""
        init = AIRTInitializer()
        assert "AIRT" in init.description or "AI Red Team" in init.description
        assert "Azure OpenAI" in init.description

    def test_required_env_vars_property(self):
        """Test the required_env_vars property."""
        init = AIRTInitializer()
        env_vars = init.required_env_vars
        
        assert isinstance(env_vars, list)
        assert "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT" in env_vars
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY" in env_vars
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2" in env_vars
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2" in env_vars
        assert "AZURE_CONTENT_SAFETY_API_ENDPOINT" in env_vars
        assert "AZURE_CONTENT_SAFETY_API_KEY" in env_vars

    def test_execution_order_is_default(self):
        """Test that execution order is default (1)."""
        init = AIRTInitializer()
        assert init.execution_order == 1


class TestAIRTInitializerValidate:
    """Tests for AIRTInitializer.validate method."""

    def setup_method(self) -> None:
        """Clear environment variables before each test."""
        self.env_vars = [
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2",
            "AZURE_CONTENT_SAFETY_API_ENDPOINT",
            "AZURE_CONTENT_SAFETY_API_KEY",
        ]
        for var in self.env_vars:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self) -> None:
        """Clean up environment variables after each test."""
        for var in self.env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_validate_passes_with_all_env_vars(self):
        """Test that validate passes when all env vars are set."""
        # Set all required env vars
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test2.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "safety_key"

        init = AIRTInitializer()
        # Should not raise any errors
        init.validate()

    def test_validate_fails_with_missing_endpoint(self):
        """Test that validate fails when converter endpoint is missing."""
        # Set all except one
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test2.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "safety_key"

        init = AIRTInitializer()
        with pytest.raises(ValueError, match="AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"):
            init.validate()

    def test_validate_fails_with_missing_key(self):
        """Test that validate fails when converter key is missing."""
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test2.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "safety_key"

        init = AIRTInitializer()
        with pytest.raises(ValueError, match="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"):
            init.validate()

    def test_validate_fails_with_missing_scorer_endpoint(self):
        """Test that validate fails when scorer endpoint is missing."""
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "safety_key"

        init = AIRTInitializer()
        with pytest.raises(ValueError, match="AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"):
            init.validate()

    def test_validate_fails_with_missing_content_safety(self):
        """Test that validate fails when content safety vars are missing."""
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test2.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"

        init = AIRTInitializer()
        with pytest.raises(ValueError, match="AZURE_CONTENT_SAFETY"):
            init.validate()

    def test_validate_fails_with_multiple_missing_vars(self):
        """Test that validate reports all missing vars."""
        # Only set one var
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"

        init = AIRTInitializer()
        with pytest.raises(ValueError) as exc_info:
            init.validate()
        
        # Should mention multiple missing vars
        error_msg = str(exc_info.value)
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY" in error_msg

    def test_validate_fails_with_no_env_vars(self):
        """Test that validate fails when no env vars are set."""
        init = AIRTInitializer()
        with pytest.raises(ValueError) as exc_info:
            init.validate()
        
        # Should mention environment variables in error
        error_msg = str(exc_info.value)
        assert "environment variables" in error_msg.lower()


class TestAIRTInitializerInitialize:
    """Tests for AIRTInitializer.initialize method."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Set up required env vars
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test2.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "safety_key"
        
        # Clean up globals
        for attr in ["default_converter_target", "default_harm_scorer", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        # Clean up env vars
        env_vars = [
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2",
            "AZURE_CONTENT_SAFETY_API_ENDPOINT",
            "AZURE_CONTENT_SAFETY_API_KEY",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Clean up globals
        for attr in ["default_converter_target", "default_harm_scorer", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def test_initialize_runs_without_error(self):
        """Test that initialize runs without errors."""
        init = AIRTInitializer()
        # Should not raise any errors
        init.initialize()

    def test_initialize_sets_default_converter_target(self):
        """Test that initialize sets default_converter_target global variable."""
        init = AIRTInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "default_converter_target")
        converter_target = sys.modules["__main__"].default_converter_target  # type: ignore
        
        # Should be an OpenAIChatTarget
        from pyrit.prompt_target import OpenAIChatTarget

        assert isinstance(converter_target, OpenAIChatTarget)

    def test_initialize_sets_converter_target_with_correct_endpoint(self):
        """Test that converter target uses the correct endpoint."""
        init = AIRTInitializer()
        init.initialize()

        converter_target = sys.modules["__main__"].default_converter_target  # type: ignore
        assert converter_target.endpoint == "https://test.openai.azure.com"

    def test_initialize_sets_converter_target_with_correct_temperature(self):
        """Test that converter target has correct temperature."""
        init = AIRTInitializer()
        init.initialize()

        converter_target = sys.modules["__main__"].default_converter_target  # type: ignore
        # AIRT config uses temperature 1.1 for converters
        assert converter_target.temperature == 1.1

    def test_initialize_sets_default_harm_scorer(self):
        """Test that initialize sets default_harm_scorer global variable."""
        init = AIRTInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "default_harm_scorer")
        scorer = sys.modules["__main__"].default_harm_scorer  # type: ignore
        
        # Should be a TrueFalseCompositeScorer
        from pyrit.score import TrueFalseCompositeScorer

        assert isinstance(scorer, TrueFalseCompositeScorer)

    def test_initialize_sets_default_objective_scorer(self):
        """Test that initialize sets default_objective_scorer global variable."""
        init = AIRTInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "default_objective_scorer")
        scorer = sys.modules["__main__"].default_objective_scorer  # type: ignore
        
        # Should be a TrueFalseCompositeScorer
        from pyrit.score import TrueFalseCompositeScorer

        assert isinstance(scorer, TrueFalseCompositeScorer)

    def test_initialize_sets_adversarial_config(self):
        """Test that initialize sets adversarial_config global variable."""
        init = AIRTInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "adversarial_config")
        from pyrit.executor.attack import AttackAdversarialConfig

        config = sys.modules["__main__"].adversarial_config  # type: ignore
        assert isinstance(config, AttackAdversarialConfig)

    def test_initialize_sets_adversarial_target_with_correct_temperature(self):
        """Test that adversarial target has correct temperature."""
        init = AIRTInitializer()
        init.initialize()

        config = sys.modules["__main__"].adversarial_config  # type: ignore
        # AIRT config uses temperature 1.2 for adversarial targets
        assert config.target.temperature == 1.2

    def test_initialize_sets_default_values_for_prompt_converter(self):
        """Test that default values are set for PromptConverter."""
        init = AIRTInitializer()
        init.initialize()

        from pyrit.prompt_converter import PromptConverter

        registry = get_global_default_values()
        
        # Should have a default value for PromptConverter.converter_target
        defaults = registry._default_values
        found = False
        for scope in defaults:
            if scope.class_type == PromptConverter and scope.parameter_name == "converter_target":
                found = True
                break
        
        assert found, "Default value for PromptConverter.converter_target not set"

    def test_initialize_sets_default_values_for_all_attack_classes(self):
        """Test that default values are set for all attack classes."""
        init = AIRTInitializer()
        init.initialize()

        from pyrit.executor.attack import (
            CrescendoAttack,
            PromptSendingAttack,
            RedTeamingAttack,
            TreeOfAttacksWithPruningAttack,
        )

        registry = get_global_default_values()
        defaults = registry._default_values

        # Should have defaults for attack_scoring_config for all attack classes
        attack_classes = [PromptSendingAttack, CrescendoAttack, RedTeamingAttack, TreeOfAttacksWithPruningAttack]
        
        for attack_class in attack_classes:
            found = False
            for scope in defaults:
                if scope.class_type == attack_class and scope.parameter_name == "attack_scoring_config":
                    found = True
                    break
            assert found, f"Default value for {attack_class.__name__}.attack_scoring_config not set"

    def test_initialize_sets_default_adversarial_config_for_all_attacks(self):
        """Test that default adversarial config is set for all attack classes."""
        init = AIRTInitializer()
        init.initialize()

        from pyrit.executor.attack import (
            CrescendoAttack,
            PromptSendingAttack,
            RedTeamingAttack,
            TreeOfAttacksWithPruningAttack,
        )

        registry = get_global_default_values()
        defaults = registry._default_values

        # Should have defaults for attack_adversarial_config for all attack classes
        attack_classes = [PromptSendingAttack, CrescendoAttack, RedTeamingAttack, TreeOfAttacksWithPruningAttack]
        
        for attack_class in attack_classes:
            found = False
            for scope in defaults:
                if scope.class_type == attack_class and scope.parameter_name == "attack_adversarial_config":
                    found = True
                    break
            assert found, f"Default value for {attack_class.__name__}.attack_adversarial_config not set"


class TestAIRTInitializerGetInfo:
    """Tests for AIRTInitializer.get_info method."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Set up memory for get_info
        from pyrit.memory import CentralMemory, SQLiteMemory

        memory = SQLiteMemory(db_path=":memory:")
        CentralMemory.set_memory_instance(memory)
        
        # Set up required env vars
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"] = "test_key"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"] = "https://test2.openai.azure.com"
        os.environ["AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"] = "test_key2"
        os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"] = "https://test.cognitiveservices.azure.com"
        os.environ["AZURE_CONTENT_SAFETY_API_KEY"] = "safety_key"

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        from pyrit.memory import CentralMemory

        CentralMemory.set_memory_instance(None)  # type: ignore
        
        # Clean up env vars
        env_vars = [
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2",
            "AZURE_CONTENT_SAFETY_API_ENDPOINT",
            "AZURE_CONTENT_SAFETY_API_KEY",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_get_info_returns_dict(self):
        """Test that get_info returns a dictionary."""
        info = AIRTInitializer.get_info()
        assert isinstance(info, dict)

    def test_get_info_has_basic_fields(self):
        """Test that get_info has name, description, and class."""
        info = AIRTInitializer.get_info()
        
        assert "name" in info
        assert "description" in info
        assert "class" in info
        assert info["name"] == "AIRT Default Configuration"
        assert info["class"] == "AIRTInitializer"

    def test_get_info_has_required_env_vars(self):
        """Test that get_info includes all required environment variables."""
        info = AIRTInitializer.get_info()
        
        assert "required_env_vars" in info
        assert "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT" in info["required_env_vars"]
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY" in info["required_env_vars"]
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2" in info["required_env_vars"]
        assert "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2" in info["required_env_vars"]
        assert "AZURE_CONTENT_SAFETY_API_ENDPOINT" in info["required_env_vars"]
        assert "AZURE_CONTENT_SAFETY_API_KEY" in info["required_env_vars"]

    def test_get_info_has_execution_order(self):
        """Test that get_info includes execution order."""
        info = AIRTInitializer.get_info()
        
        assert "execution_order" in info
        assert info["execution_order"] == 1

    def test_get_info_has_defaults_info(self):
        """Test that get_info includes default values information."""
        info = AIRTInitializer.get_info()
        
        assert "default_values" in info
        assert "global_variables" in info
