# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the setup.AttackFactory."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyrit.prompt_target import PromptTarget
from pyrit.setup import AttackFactory, ConfigurationPaths


@pytest.fixture
def mock_target():
    """Create a mock PromptTarget for testing."""
    return MagicMock(spec=PromptTarget)


@pytest.mark.usefixtures("patch_central_database")
class TestAttackFactory:
    """Tests for the AttackFactory class."""

    def test_create_attack_with_ascii_art_config(self, mock_target: PromptTarget):
        """Test creating an attack from the ascii_art config file."""
        attack = AttackFactory.create_attack(
            config_path=ConfigurationPaths.attack.foundry.ascii_art,
            objective_target=mock_target,
        )
        
        from pyrit.executor.attack.single_turn import PromptSendingAttack
        assert isinstance(attack, PromptSendingAttack)
        assert attack._objective_target == mock_target
        assert attack._request_converters is not None

    def test_create_attack_with_ansi_config(self, mock_target: PromptTarget):
        """Test creating an attack from the ansi_attack config file."""
        attack = AttackFactory.create_attack(
            config_path=ConfigurationPaths.attack.foundry.ansi_attack,
            objective_target=mock_target,
        )
        
        from pyrit.executor.attack.single_turn import PromptSendingAttack
        assert isinstance(attack, PromptSendingAttack)
        assert attack._objective_target == mock_target
        assert attack._request_converters is not None

    def test_create_attack_with_crescendo_config(self, mock_target: PromptTarget):
        """Test creating an attack from the crescendo config file."""
        # Crescendo requires adversarial_config, so we need to provide it
        from pyrit.executor.attack import AttackAdversarialConfig
        from pyrit.prompt_target import OpenAIChatTarget
        
        adversarial_chat = MagicMock(spec=OpenAIChatTarget)
        adversarial_config = AttackAdversarialConfig(target=adversarial_chat)
        
        attack = AttackFactory.create_attack(
            config_path=ConfigurationPaths.attack.foundry.crescendo,
            objective_target=mock_target,
            attack_adversarial_config=adversarial_config,
        )
        
        from pyrit.executor.attack.multi_turn import CrescendoAttack
        assert isinstance(attack, CrescendoAttack)
        assert attack._objective_target == mock_target
        assert attack._max_turns == 3
        assert attack._max_backtracks == 2

    def test_create_attack_with_tense_config_no_defaults(self, mock_target: PromptTarget):
        """Test that tense config raises error when defaults aren't set."""
        # Tense config requires converter_target default to be set
        # Test that we get a helpful error when defaults aren't set
        with pytest.raises(ValueError, match="requires default values to be set"):
            AttackFactory.create_attack(
                config_path=ConfigurationPaths.attack.foundry.tense,
                objective_target=mock_target,
            )

    def test_create_attack_with_override_params(self, mock_target: PromptTarget):
        """Test that override parameters work correctly."""
        from pyrit.executor.attack import AttackConverterConfig
        from pyrit.prompt_normalizer import PromptConverterConfiguration
        
        custom_config = AttackConverterConfig(
            request_converters=PromptConverterConfiguration.from_converters(converters=[])
        )
        
        attack = AttackFactory.create_attack(
            config_path=ConfigurationPaths.attack.foundry.ascii_art,
            objective_target=mock_target,
            attack_converter_config=custom_config,
        )
        
        from pyrit.executor.attack.single_turn import PromptSendingAttack
        assert isinstance(attack, PromptSendingAttack)
        assert attack._request_converters == custom_config.request_converters

    def test_create_attack_nonexistent_file(self, mock_target: PromptTarget):
        """Test that FileNotFoundError is raised for nonexistent config file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            AttackFactory.create_attack(
                config_path="nonexistent_config.py",
                objective_target=mock_target,
            )

    def test_create_attack_invalid_config_no_attack_config(self, mock_target: PromptTarget):
        """Test that AttributeError is raised when config file doesn't define attack_config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Empty config file\n")
            f.write("some_variable = 42\n")
            temp_path = f.name
        
        try:
            with pytest.raises(AttributeError, match="must define an 'attack_config' dictionary"):
                AttackFactory.create_attack(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_attack_invalid_config_not_dict(self, mock_target: PromptTarget):
        """Test that ValueError is raised when attack_config is not a dictionary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("attack_config = 'not a dict'\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must be a dictionary"):
                AttackFactory.create_attack(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_attack_invalid_config_no_attack_type(self, mock_target: PromptTarget):
        """Test that ValueError is raised when attack_config doesn't specify attack_type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("attack_config = {'some_param': 'value'}\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must define 'attack_type'"):
                AttackFactory.create_attack(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_attack_unsupported_attack_type(self, mock_target: PromptTarget):
        """Test that ValueError is raised for unsupported attack types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("attack_config = {'attack_type': 'InvalidAttackType'}\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported attack type"):
                AttackFactory.create_attack(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_attack_returns_objects_with_execute_async(self, mock_target: PromptTarget):
        """Test that created attacks have an execute_async method."""
        attack = AttackFactory.create_attack(
            config_path=ConfigurationPaths.attack.foundry.ascii_art,
            objective_target=mock_target,
        )
        assert hasattr(attack, "execute_async")
        assert callable(attack.execute_async)

    def test_create_attack_from_config_function(self, mock_target: PromptTarget):
        """Test the convenience function create_attack_from_config."""
        from pyrit.setup import create_attack_from_config
        
        attack = create_attack_from_config(
            config_path=ConfigurationPaths.attack.foundry.ascii_art,
            objective_target=mock_target,
        )
        
        from pyrit.executor.attack.single_turn import PromptSendingAttack
        assert isinstance(attack, PromptSendingAttack)
        assert attack._objective_target == mock_target
