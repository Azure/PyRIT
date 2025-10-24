# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import pytest

from pyrit.common.apply_defaults import get_global_default_values, reset_default_values
from pyrit.setup.initializers.simple import SimpleInitializer


class TestSimpleInitializer:
    """Tests for SimpleInitializer class."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Clean up globals
        for attr in ["default_converter_target", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        for attr in ["default_converter_target", "default_objective_scorer", "adversarial_config"]:
            if hasattr(sys.modules["__main__"], attr):
                delattr(sys.modules["__main__"], attr)

    def test_simple_initializer_can_be_created(self):
        """Test that SimpleInitializer can be instantiated."""
        init = SimpleInitializer()
        assert init is not None

    def test_name_property(self):
        """Test the name property."""
        init = SimpleInitializer()
        assert init.name == "Simple Complete Configuration"

    def test_description_property(self):
        """Test the description property."""
        init = SimpleInitializer()
        assert "simple" in init.description.lower()
        assert "openai" in init.description.lower()

    def test_required_env_vars_property(self):
        """Test the required_env_vars property."""
        init = SimpleInitializer()
        env_vars = init.required_env_vars

        assert isinstance(env_vars, list)
        assert "OPENAI_CHAT_ENDPOINT" in env_vars
        assert "OPENAI_CHAT_KEY" in env_vars

    def test_execution_order_is_default(self):
        """Test that execution order is default (1)."""
        init = SimpleInitializer()
        assert init.execution_order == 1

    def test_validate_does_not_raise(self):
        """Test that validate method does not raise errors."""
        init = SimpleInitializer()
        # Should not raise any errors (simple config has no validation)
        init.validate()


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

    def test_initialize_sets_default_converter_target(self):
        """Test that initialize sets default_converter_target global variable."""
        init = SimpleInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "default_converter_target")
        converter_target = sys.modules["__main__"].default_converter_target  # type: ignore

        # Should be an OpenAIChatTarget
        from pyrit.prompt_target import OpenAIChatTarget

        assert isinstance(converter_target, OpenAIChatTarget)

    def test_initialize_sets_converter_target_with_correct_temperature(self):
        """Test that converter target has correct temperature."""
        init = SimpleInitializer()
        init.initialize()

        converter_target = sys.modules["__main__"].default_converter_target  # type: ignore
        # Simple config uses temperature 1.2 for converters
        assert converter_target._temperature == 1.2

    def test_initialize_sets_default_objective_scorer(self):
        """Test that initialize sets default_objective_scorer global variable."""
        init = SimpleInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "default_objective_scorer")
        scorer = sys.modules["__main__"].default_objective_scorer  # type: ignore

        # Should be a TrueFalseCompositeScorer
        from pyrit.score import TrueFalseCompositeScorer

        assert isinstance(scorer, TrueFalseCompositeScorer)

    def test_initialize_sets_adversarial_config(self):
        """Test that initialize sets adversarial_config global variable."""
        init = SimpleInitializer()
        init.initialize()

        assert hasattr(sys.modules["__main__"], "adversarial_config")
        config = sys.modules["__main__"].adversarial_config  # type: ignore

        # Should be an AttackAdversarialConfig
        from pyrit.executor.attack import AttackAdversarialConfig

        assert isinstance(config, AttackAdversarialConfig)

    def test_initialize_sets_adversarial_target_with_correct_temperature(self):
        """Test that adversarial target has correct temperature."""
        init = SimpleInitializer()
        init.initialize()

        config = sys.modules["__main__"].adversarial_config  # type: ignore
        # Simple config uses temperature 1.3 for adversarial targets
        assert config.target._temperature == 1.3

    def test_initialize_sets_default_values_for_prompt_converter(self):
        """Test that default values are set for PromptConverter."""
        init = SimpleInitializer()
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

    def test_initialize_sets_default_values_for_attack_classes(self):
        """Test that default values are set for attack classes."""
        init = SimpleInitializer()
        init.initialize()

        from pyrit.executor.attack import (
            CrescendoAttack,
            PromptSendingAttack,
            RedTeamingAttack,
            TreeOfAttacksWithPruningAttack,
        )

        registry = get_global_default_values()
        defaults = registry._default_values

        # Should have defaults for attack_scoring_config
        attack_classes = [PromptSendingAttack, CrescendoAttack, RedTeamingAttack, TreeOfAttacksWithPruningAttack]

        for attack_class in attack_classes:
            found = False
            for scope in defaults:
                if scope.class_type == attack_class and scope.parameter_name == "attack_scoring_config":
                    found = True
                    break
            assert found, f"Default value for {attack_class.__name__}.attack_scoring_config not set"

    def test_initialize_sets_default_adversarial_config_for_crescendo(self):
        """Test that default adversarial config is set for CrescendoAttack."""
        init = SimpleInitializer()
        init.initialize()

        from pyrit.executor.attack import CrescendoAttack

        registry = get_global_default_values()
        defaults = registry._default_values

        # Should have default for CrescendoAttack.attack_adversarial_config
        found = False
        for scope in defaults:
            if scope.class_type == CrescendoAttack and scope.parameter_name == "attack_adversarial_config":
                found = True
                break

        assert found, "Default value for CrescendoAttack.attack_adversarial_config not set"


class TestSimpleInitializerGetInfo:
    """Tests for SimpleInitializer.get_info method."""

    def setup_method(self) -> None:
        """Set up before each test."""
        reset_default_values()
        # Set up memory for get_info
        from pyrit.memory import CentralMemory, SQLiteMemory

        memory = SQLiteMemory(db_path=":memory:")
        CentralMemory.set_memory_instance(memory)

        # Set up required env vars
        os.environ["OPENAI_CHAT_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["OPENAI_CHAT_KEY"] = "test_key"

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_default_values()
        from pyrit.memory import CentralMemory

        CentralMemory.set_memory_instance(None)  # type: ignore

        # Clean up env vars
        for var in ["OPENAI_CHAT_ENDPOINT", "OPENAI_CHAT_KEY"]:
            if var in os.environ:
                del os.environ[var]

    def test_get_info_returns_dict(self):
        """Test that get_info returns a dictionary."""
        info = SimpleInitializer.get_info()
        assert isinstance(info, dict)

    def test_get_info_has_basic_fields(self):
        """Test that get_info has name, description, and class."""
        info = SimpleInitializer.get_info()

        assert "name" in info
        assert "description" in info
        assert "class" in info
        assert info["name"] == "Simple Complete Configuration"
        assert info["class"] == "SimpleInitializer"

    def test_get_info_has_required_env_vars(self):
        """Test that get_info includes required environment variables."""
        info = SimpleInitializer.get_info()

        assert "required_env_vars" in info
        assert "OPENAI_CHAT_ENDPOINT" in info["required_env_vars"]
        assert "OPENAI_CHAT_KEY" in info["required_env_vars"]

    def test_get_info_has_execution_order(self):
        """Test that get_info includes execution order."""
        info = SimpleInitializer.get_info()

        assert "execution_order" in info
        assert info["execution_order"] == 1

    def test_get_info_has_defaults_info(self):
        """Test that get_info includes default values information."""
        info = SimpleInitializer.get_info()

        assert "default_values" in info
        assert "global_variables" in info
