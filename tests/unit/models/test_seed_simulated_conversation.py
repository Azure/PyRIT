# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the SeedSimulatedConversation class."""

import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models.seeds import SeedSimulatedConversation


@pytest.fixture
def basic_simulated_conversation():
    """Create a basic SeedSimulatedConversation for testing."""
    return SeedSimulatedConversation(
        value="",  # Will be set by __post_init__
        num_turns=3,
        adversarial_system_prompt="You are a red teaming agent.",
        simulated_target_system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def minimal_simulated_conversation():
    """Create a minimal SeedSimulatedConversation with only required fields."""
    return SeedSimulatedConversation(
        value="",  # Will be set by __post_init__
        num_turns=2,
        adversarial_system_prompt="Adversarial prompt",
    )


class TestSeedSimulatedConversationInit:
    """Tests for SeedSimulatedConversation initialization."""

    def test_init_with_all_parameters(self, basic_simulated_conversation):
        """Test initialization with all parameters."""
        assert basic_simulated_conversation.num_turns == 3
        assert basic_simulated_conversation.adversarial_system_prompt == "You are a red teaming agent."
        assert basic_simulated_conversation.simulated_target_system_prompt == "You are a helpful assistant."
        assert basic_simulated_conversation.data_type == "text"
        assert isinstance(basic_simulated_conversation.id, uuid.UUID)

    def test_init_with_minimal_parameters(self, minimal_simulated_conversation):
        """Test initialization with only required parameters."""
        assert minimal_simulated_conversation.num_turns == 2
        assert minimal_simulated_conversation.adversarial_system_prompt == "Adversarial prompt"
        assert minimal_simulated_conversation.simulated_target_system_prompt is None

    def test_init_default_num_turns(self):
        """Test that default num_turns is 3."""
        conv = SeedSimulatedConversation(
            value="",
            adversarial_system_prompt="test",
        )
        assert conv.num_turns == 3

    def test_init_invalid_num_turns_zero_raises_error(self):
        """Test that num_turns=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_turns must be a positive integer"):
            SeedSimulatedConversation(
                value="",
                num_turns=0,
                adversarial_system_prompt="test",
            )

    def test_init_invalid_num_turns_negative_raises_error(self):
        """Test that negative num_turns raises ValueError."""
        with pytest.raises(ValueError, match="num_turns must be a positive integer"):
            SeedSimulatedConversation(
                value="",
                num_turns=-1,
                adversarial_system_prompt="test",
            )

    def test_init_sets_data_type_to_text(self, basic_simulated_conversation):
        """Test that data_type is always set to 'text'."""
        assert basic_simulated_conversation.data_type == "text"

    def test_init_generates_json_value(self, basic_simulated_conversation):
        """Test that value is set to a JSON serialization of config."""
        value = json.loads(basic_simulated_conversation.value)
        assert value["num_turns"] == 3
        assert value["adversarial_system_prompt"] == "You are a red teaming agent."
        assert value["simulated_target_system_prompt"] == "You are a helpful assistant."
        assert "pyrit_version" in value

    def test_init_value_is_deterministic(self):
        """Test that the same config produces the same value."""
        conv1 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test",
            simulated_target_system_prompt="target",
        )
        conv2 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test",
            simulated_target_system_prompt="target",
        )
        assert conv1.value == conv2.value


class TestSeedSimulatedConversationFromDict:
    """Tests for SeedSimulatedConversation.from_dict method."""

    def test_from_dict_with_direct_strings(self):
        """Test from_dict with direct string values."""
        data = {
            "num_turns": 5,
            "adversarial_system_prompt": "You are adversarial.",
            "simulated_target_system_prompt": "You are a target.",
        }
        conv = SeedSimulatedConversation.from_dict(data)

        assert conv.num_turns == 5
        assert conv.adversarial_system_prompt == "You are adversarial."
        assert conv.simulated_target_system_prompt == "You are a target."

    def test_from_dict_without_simulated_target_prompt(self):
        """Test from_dict without simulated_target_system_prompt."""
        data = {
            "num_turns": 3,
            "adversarial_system_prompt": "You are adversarial.",
        }
        conv = SeedSimulatedConversation.from_dict(data)

        assert conv.num_turns == 3
        assert conv.adversarial_system_prompt == "You are adversarial."
        assert conv.simulated_target_system_prompt is None

    def test_from_dict_default_num_turns(self):
        """Test from_dict uses default num_turns when not specified."""
        data = {
            "adversarial_system_prompt": "You are adversarial.",
        }
        conv = SeedSimulatedConversation.from_dict(data)

        assert conv.num_turns == 3

    def test_from_dict_with_paths_calls_from_yaml_paths(self):
        """Test that from_dict with paths delegates to from_yaml_paths."""
        with patch.object(SeedSimulatedConversation, "from_yaml_paths") as mock_from_yaml:
            mock_conv = MagicMock(spec=SeedSimulatedConversation)
            mock_from_yaml.return_value = mock_conv

            data = {
                "num_turns": 4,
                "adversarial_system_prompt_path": "path/to/adversarial.yaml",
                "simulated_target_system_prompt_path": "path/to/simulated.yaml",
            }
            result = SeedSimulatedConversation.from_dict(data)

            mock_from_yaml.assert_called_once_with(
                num_turns=4,
                adversarial_system_prompt_path="path/to/adversarial.yaml",
                simulated_target_system_prompt_path="path/to/simulated.yaml",
            )
            assert result == mock_conv


class TestSeedSimulatedConversationFromYamlPaths:
    """Tests for SeedSimulatedConversation.from_yaml_paths method."""

    def test_from_yaml_paths_loads_adversarial_prompt(self, tmp_path):
        """Test that from_yaml_paths correctly loads adversarial prompt from file."""
        # Create a temporary YAML file
        yaml_content = """
value: "You are a red team agent."
data_type: text
"""
        adv_path = tmp_path / "adversarial.yaml"
        adv_path.write_text(yaml_content)

        conv = SeedSimulatedConversation.from_yaml_paths(
            num_turns=5,
            adversarial_system_prompt_path=adv_path,
        )

        assert conv.num_turns == 5
        assert conv.adversarial_system_prompt == "You are a red team agent."
        assert conv.simulated_target_system_prompt is None

    def test_from_yaml_paths_loads_both_prompts(self, tmp_path):
        """Test that from_yaml_paths loads both adversarial and simulated prompts."""
        # Create temporary YAML files
        adv_content = """
value: "You are adversarial."
data_type: text
"""
        sim_content = """
value: "You are a simulated target."
data_type: text
"""
        adv_path = tmp_path / "adversarial.yaml"
        adv_path.write_text(adv_content)
        sim_path = tmp_path / "simulated.yaml"
        sim_path.write_text(sim_content)

        conv = SeedSimulatedConversation.from_yaml_paths(
            num_turns=2,
            adversarial_system_prompt_path=adv_path,
            simulated_target_system_prompt_path=sim_path,
        )

        assert conv.num_turns == 2
        assert conv.adversarial_system_prompt == "You are adversarial."
        assert conv.simulated_target_system_prompt == "You are a simulated target."

    def test_from_yaml_paths_default_num_turns(self, tmp_path):
        """Test that from_yaml_paths uses default num_turns."""
        yaml_content = """
value: "Test prompt"
data_type: text
"""
        adv_path = tmp_path / "adversarial.yaml"
        adv_path.write_text(yaml_content)

        conv = SeedSimulatedConversation.from_yaml_paths(
            adversarial_system_prompt_path=adv_path,
        )

        assert conv.num_turns == 3


class TestSeedSimulatedConversationGetIdentifier:
    """Tests for SeedSimulatedConversation.get_identifier method."""

    def test_get_identifier_returns_correct_structure(self, basic_simulated_conversation):
        """Test that get_identifier returns the expected structure."""
        identifier = basic_simulated_conversation.get_identifier()

        assert identifier["__type__"] == "SeedSimulatedConversation"
        assert identifier["num_turns"] == 3
        assert "adversarial_system_prompt_hash" in identifier
        assert "simulated_target_system_prompt_hash" in identifier
        assert "pyrit_version" in identifier

    def test_get_identifier_hashes_prompts(self, basic_simulated_conversation):
        """Test that prompts are hashed in the identifier."""
        identifier = basic_simulated_conversation.get_identifier()

        assert identifier["adversarial_system_prompt_hash"].startswith("sha256:")
        assert identifier["simulated_target_system_prompt_hash"].startswith("sha256:")

    def test_get_identifier_none_prompt_is_none(self, minimal_simulated_conversation):
        """Test that None prompts return None in identifier."""
        identifier = minimal_simulated_conversation.get_identifier()

        assert identifier["simulated_target_system_prompt_hash"] is None


class TestSeedSimulatedConversationComputeHash:
    """Tests for SeedSimulatedConversation.compute_hash method."""

    def test_compute_hash_returns_sha256(self, basic_simulated_conversation):
        """Test that compute_hash returns a valid SHA256 hash."""
        hash_value = basic_simulated_conversation.compute_hash()

        # SHA256 hash is 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_compute_hash_is_deterministic(self):
        """Test that the same config produces the same hash."""
        conv1 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test",
            simulated_target_system_prompt="target",
        )
        conv2 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test",
            simulated_target_system_prompt="target",
        )

        assert conv1.compute_hash() == conv2.compute_hash()

    def test_compute_hash_differs_for_different_configs(self):
        """Test that different configs produce different hashes."""
        conv1 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test1",
        )
        conv2 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test2",
        )

        assert conv1.compute_hash() != conv2.compute_hash()

    def test_compute_hash_differs_for_different_num_turns(self):
        """Test that different num_turns produces different hash."""
        conv1 = SeedSimulatedConversation(
            value="",
            num_turns=3,
            adversarial_system_prompt="test",
        )
        conv2 = SeedSimulatedConversation(
            value="",
            num_turns=5,
            adversarial_system_prompt="test",
        )

        assert conv1.compute_hash() != conv2.compute_hash()


class TestSeedSimulatedConversationHashString:
    """Tests for SeedSimulatedConversation._hash_string method."""

    def test_hash_string_returns_sha256_prefix(self):
        """Test that _hash_string returns sha256: prefixed hash."""
        result = SeedSimulatedConversation._hash_string("test string")

        assert result.startswith("sha256:")
        assert len(result) == 23  # "sha256:" (7) + 16 hex chars

    def test_hash_string_none_returns_none(self):
        """Test that _hash_string returns None for None input."""
        result = SeedSimulatedConversation._hash_string(None)

        assert result is None

    def test_hash_string_is_deterministic(self):
        """Test that the same string produces the same hash."""
        result1 = SeedSimulatedConversation._hash_string("test")
        result2 = SeedSimulatedConversation._hash_string("test")

        assert result1 == result2

    def test_hash_string_differs_for_different_inputs(self):
        """Test that different strings produce different hashes."""
        result1 = SeedSimulatedConversation._hash_string("test1")
        result2 = SeedSimulatedConversation._hash_string("test2")

        assert result1 != result2


class TestSeedSimulatedConversationRepr:
    """Tests for SeedSimulatedConversation.__repr__ method."""

    def test_repr_with_all_prompts(self, basic_simulated_conversation):
        """Test __repr__ with all prompts set."""
        repr_str = repr(basic_simulated_conversation)

        assert "SeedSimulatedConversation" in repr_str
        assert "num_turns=3" in repr_str
        assert "adv_prompt=yes" in repr_str
        assert "sim_prompt=yes" in repr_str

    def test_repr_without_simulated_prompt(self, minimal_simulated_conversation):
        """Test __repr__ without simulated target prompt."""
        repr_str = repr(minimal_simulated_conversation)

        assert "SeedSimulatedConversation" in repr_str
        assert "num_turns=2" in repr_str
        assert "adv_prompt=yes" in repr_str
        assert "sim_prompt=no" in repr_str

    def test_repr_without_adversarial_prompt(self):
        """Test __repr__ without adversarial prompt."""
        conv = SeedSimulatedConversation(
            value="",
            num_turns=4,
        )
        repr_str = repr(conv)

        assert "num_turns=4" in repr_str
        assert "adv_prompt=no" in repr_str
        assert "sim_prompt=no" in repr_str


class TestSeedSimulatedConversationIntegration:
    """Integration tests for SeedSimulatedConversation."""

    def test_seed_properties_inherited(self, basic_simulated_conversation):
        """Test that Seed base class properties work correctly."""
        # Check inherited properties
        assert isinstance(basic_simulated_conversation.id, uuid.UUID)
        # prompt_group_id is None by default for standalone seeds
        # it gets set when added to a SeedGroup

    def test_can_be_used_in_seed_attack_group(self, basic_simulated_conversation):
        """Test that SeedSimulatedConversation works with SeedAttackGroup."""
        from pyrit.models.seeds import SeedAttackGroup, SeedObjective

        group = SeedAttackGroup(
            seeds=[
                SeedObjective(value="Test objective"),
                basic_simulated_conversation,
            ]
        )

        assert group.has_simulated_conversation
        assert group.simulated_conversation_config == basic_simulated_conversation

    def test_from_dict_via_seed_attack_group(self):
        """Test creating SeedSimulatedConversation via SeedAttackGroup from dict."""
        from pyrit.models.seeds import SeedAttackGroup

        group = SeedAttackGroup(
            seeds=[
                {"value": "Test objective", "is_objective": True},
                {
                    "is_simulated_conversation": True,
                    "num_turns": 4,
                    "adversarial_system_prompt": "Adversarial",
                    "simulated_target_system_prompt": "Target",
                },
            ]
        )

        assert group.has_simulated_conversation
        config = group.simulated_conversation_config
        assert config.num_turns == 4
        assert config.adversarial_system_prompt == "Adversarial"
        assert config.simulated_target_system_prompt == "Target"
