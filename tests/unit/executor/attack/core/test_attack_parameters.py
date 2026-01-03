# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dataclasses
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack.core.attack_parameters import (
    AttackParameters,
    _build_params_from_seed_group_async,
)
from pyrit.models import Message, MessagePiece, SeedAttackGroup, SeedObjective, SeedPrompt
from pyrit.models.literals import ChatMessageRole


def _make_message(role: ChatMessageRole, content: str) -> Message:
    """Helper to create a Message from role and content."""
    return Message(message_pieces=[MessagePiece(role=role, original_value=content)])


class TestFromSeedGroupAsync:
    """Tests for AttackParameters.from_seed_group_async."""

    @pytest.fixture
    def seed_objective(self) -> SeedObjective:
        """Create a basic seed objective."""
        return SeedObjective(value="Test objective")

    @pytest.fixture
    def seed_group_with_objective(self, seed_objective: SeedObjective) -> SeedAttackGroup:
        """Create a SeedAttackGroup with just an objective."""
        return SeedAttackGroup(seeds=[seed_objective])

    async def test_extracts_objective_from_seed_group(
        self, seed_group_with_objective: SeedAttackGroup
    ) -> None:
        """Test that objective is correctly extracted from seed group."""
        params = await AttackParameters.from_seed_group_async(seed_group_with_objective)

        assert params.objective == "Test objective"

    async def test_raises_when_no_objective(self) -> None:
        """Test that ValueError is raised when seed group has no objective."""
        # Create a seed group with only a prompt (no objective)
        prompt = SeedPrompt(value="Some prompt", data_type="text", role="user")
        seed_group = SeedAttackGroup(seeds=[prompt])

        with pytest.raises(ValueError, match="SeedGroup must have an objective"):
            await AttackParameters.from_seed_group_async(seed_group)

    async def test_extracts_next_message(self) -> None:
        """Test that next_message is extracted from seed group prompts."""
        objective = SeedObjective(value="Test objective")
        prompt = SeedPrompt(value="Test prompt", data_type="text", role="user")
        seed_group = SeedAttackGroup(seeds=[objective, prompt])

        params = await AttackParameters.from_seed_group_async(seed_group)

        assert params.next_message is not None
        assert params.next_message.get_value() == "Test prompt"

    async def test_extracts_prepended_conversation(self) -> None:
        """Test that prepended_conversation is extracted from multi-prompt seed groups."""
        objective = SeedObjective(value="Test objective")
        prompt1 = SeedPrompt(value="First message", data_type="text", role="user", sequence=1)
        prompt2 = SeedPrompt(value="Response", data_type="text", role="assistant", sequence=2)
        prompt3 = SeedPrompt(value="Second message", data_type="text", role="user", sequence=3)
        seed_group = SeedAttackGroup(seeds=[objective, prompt1, prompt2, prompt3])

        params = await AttackParameters.from_seed_group_async(seed_group)

        assert params.prepended_conversation is not None
        assert len(params.prepended_conversation) == 2
        assert params.next_message is not None
        assert params.next_message.get_value() == "Second message"

    async def test_applies_overrides(self, seed_group_with_objective: SeedAttackGroup) -> None:
        """Test that overrides are applied to the parameters."""
        custom_message = _make_message("user", "Override message")

        params = await AttackParameters.from_seed_group_async(
            seed_group_with_objective,
            next_message=custom_message,
        )

        assert params.next_message == custom_message

    async def test_rejects_invalid_overrides(self, seed_group_with_objective: SeedAttackGroup) -> None:
        """Test that invalid override fields raise ValueError."""
        with pytest.raises(ValueError, match="does not accept parameters"):
            await AttackParameters.from_seed_group_async(
                seed_group_with_objective,
                invalid_field="value",
            )


class TestFromSeedGroupAsyncWithSimulatedConversation:
    """Tests for from_seed_group_async with simulated conversation config."""

    @pytest.fixture
    def seed_objective(self) -> SeedObjective:
        """Create a basic seed objective."""
        return SeedObjective(value="Test objective")

    @pytest.fixture
    def simulated_conversation_config(self) -> MagicMock:
        """Create a mock SeedSimulatedConversation config."""
        config = MagicMock()
        config.num_turns = 3
        config.adversarial_system_prompt = "/path/to/adversarial.yaml"
        config.simulated_target_system_prompt = "/path/to/target.yaml"
        return config

    @pytest.fixture
    def seed_group_with_simulated_conv(
        self, seed_objective: SeedObjective, simulated_conversation_config: MagicMock
    ) -> SeedAttackGroup:
        """Create a SeedAttackGroup with simulated conversation config."""
        seed_group = SeedAttackGroup(seeds=[seed_objective])
        # Mock the simulated conversation properties
        seed_group._simulated_conversation_config = simulated_conversation_config
        return seed_group

    @pytest.fixture
    def mock_adversarial_chat(self) -> MagicMock:
        """Create a mock adversarial chat target."""
        return MagicMock()

    @pytest.fixture
    def mock_objective_scorer(self) -> MagicMock:
        """Create a mock objective scorer."""
        return MagicMock()

    @pytest.fixture
    def mock_simulated_result(self) -> MagicMock:
        """Create a mock simulated conversation result."""
        result = MagicMock()
        result.prepended_messages = [
            _make_message("user", "Simulated user message"),
            _make_message("assistant", "Simulated assistant response"),
        ]
        result.next_message = _make_message("user", "Final simulated message")
        return result

    async def test_raises_when_adversarial_chat_missing(
        self,
        seed_group_with_simulated_conv: SeedAttackGroup,
        mock_objective_scorer: MagicMock,
    ) -> None:
        """Test that ValueError is raised when adversarial_chat is None."""
        with pytest.raises(ValueError, match="adversarial_chat is required"):
            await AttackParameters.from_seed_group_async(
                seed_group_with_simulated_conv,
                adversarial_chat=None,
                objective_scorer=mock_objective_scorer,
            )

    async def test_raises_when_objective_scorer_missing(
        self,
        seed_group_with_simulated_conv: SeedAttackGroup,
        mock_adversarial_chat: MagicMock,
    ) -> None:
        """Test that ValueError is raised when objective_scorer is None."""
        with pytest.raises(ValueError, match="objective_scorer is required"):
            await AttackParameters.from_seed_group_async(
                seed_group_with_simulated_conv,
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=None,
            )

    async def test_raises_when_next_message_conflicts_with_simulated_conv(
        self, seed_objective: SeedObjective, simulated_conversation_config: MagicMock
    ) -> None:
        """Test that ValueError is raised when both simulated conv and next_message are set."""
        prompt = SeedPrompt(value="Static prompt", data_type="text", role="user")
        seed_group = SeedAttackGroup(seeds=[seed_objective, prompt])
        seed_group._simulated_conversation_config = simulated_conversation_config

        with pytest.raises(ValueError, match="next_message set.*mutually exclusive"):
            await AttackParameters.from_seed_group_async(seed_group)

    async def test_raises_when_multi_sequence_prompts_conflict_with_simulated_conv(
        self, seed_objective: SeedObjective, simulated_conversation_config: MagicMock
    ) -> None:
        """Test that ValueError is raised when simulated conv is set during SeedAttackGroup creation with prompts."""
        # This test verifies that the SeedAttackGroup constructor raises if we have 
        # multi-sequence prompts and simulated_conversation_config together.
        # Since validation happens at construction for multi-sequence prompts,
        # we test the from_seed_group_async path for the single prompt case.
        prompt = SeedPrompt(value="Static prompt", data_type="text", role="user")
        seed_group = SeedAttackGroup(seeds=[seed_objective, prompt])
        seed_group._simulated_conversation_config = simulated_conversation_config

        # A single prompt creates next_message, which triggers the first validation check
        with pytest.raises(ValueError, match="next_message set.*mutually exclusive"):
            await AttackParameters.from_seed_group_async(seed_group)

    @patch("pyrit.executor.attack.component.simulated_conversation.generate_simulated_conversation_async")
    async def test_generates_simulated_conversation(
        self,
        mock_generate: AsyncMock,
        seed_group_with_simulated_conv: SeedAttackGroup,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_simulated_result: MagicMock,
    ) -> None:
        """Test that simulated conversation is generated when config is present."""
        mock_generate.return_value = mock_simulated_result

        params = await AttackParameters.from_seed_group_async(
            seed_group_with_simulated_conv,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["objective"] == "Test objective"
        assert call_kwargs["adversarial_chat"] == mock_adversarial_chat
        assert call_kwargs["objective_scorer"] == mock_objective_scorer
        assert call_kwargs["num_turns"] == 3

    @patch("pyrit.executor.attack.component.simulated_conversation.generate_simulated_conversation_async")
    async def test_uses_generated_prepended_messages(
        self,
        mock_generate: AsyncMock,
        seed_group_with_simulated_conv: SeedAttackGroup,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_simulated_result: MagicMock,
    ) -> None:
        """Test that prepended_conversation comes from the generated result."""
        mock_generate.return_value = mock_simulated_result

        params = await AttackParameters.from_seed_group_async(
            seed_group_with_simulated_conv,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        assert params.prepended_conversation == mock_simulated_result.prepended_messages

    @patch("pyrit.executor.attack.component.simulated_conversation.generate_simulated_conversation_async")
    async def test_uses_generated_next_message(
        self,
        mock_generate: AsyncMock,
        seed_group_with_simulated_conv: SeedAttackGroup,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_simulated_result: MagicMock,
    ) -> None:
        """Test that next_message comes from the generated result."""
        mock_generate.return_value = mock_simulated_result

        params = await AttackParameters.from_seed_group_async(
            seed_group_with_simulated_conv,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        assert params.next_message == mock_simulated_result.next_message

    @patch("pyrit.executor.attack.component.simulated_conversation.generate_simulated_conversation_async")
    async def test_caches_result_in_seed_group(
        self,
        mock_generate: AsyncMock,
        seed_group_with_simulated_conv: SeedAttackGroup,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_simulated_result: MagicMock,
    ) -> None:
        """Test that the generated result is cached in the seed_group."""
        mock_generate.return_value = mock_simulated_result

        # Mock set_simulated_conversation_result
        seed_group_with_simulated_conv.set_simulated_conversation_result = MagicMock()

        await AttackParameters.from_seed_group_async(
            seed_group_with_simulated_conv,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        seed_group_with_simulated_conv.set_simulated_conversation_result.assert_called_once_with(
            mock_simulated_result
        )


class TestExcluding:
    """Tests for AttackParameters.excluding()."""

    def test_excluding_creates_class_without_specified_fields(self) -> None:
        """Test that excluding() creates a class without the specified fields."""
        ExcludedParams = AttackParameters.excluding("next_message", "prepended_conversation")

        field_names = {f.name for f in dataclasses.fields(ExcludedParams)}

        assert "objective" in field_names
        assert "memory_labels" in field_names
        assert "next_message" not in field_names
        assert "prepended_conversation" not in field_names

    def test_excluding_raises_for_invalid_fields(self) -> None:
        """Test that excluding() raises ValueError for non-existent fields."""
        with pytest.raises(ValueError, match="Cannot exclude non-existent fields"):
            AttackParameters.excluding("nonexistent_field")

    async def test_excluded_class_has_from_seed_group_async(self) -> None:
        """Test that the excluded class has from_seed_group_async method."""
        ExcludedParams = AttackParameters.excluding("next_message", "prepended_conversation")

        assert hasattr(ExcludedParams, "from_seed_group_async")

    async def test_excluded_class_from_seed_group_async_works(self) -> None:
        """Test that from_seed_group_async works on excluded class."""
        ExcludedParams = AttackParameters.excluding("next_message", "prepended_conversation")
        objective = SeedObjective(value="Test objective")
        seed_group = SeedAttackGroup(seeds=[objective])

        params = await ExcludedParams.from_seed_group_async(seed_group)

        assert params.objective == "Test objective"

    async def test_excluded_class_rejects_excluded_field_overrides(self) -> None:
        """Test that from_seed_group_async rejects overrides for excluded fields."""
        ExcludedParams = AttackParameters.excluding("next_message")
        objective = SeedObjective(value="Test objective")
        seed_group = SeedAttackGroup(seeds=[objective])

        with pytest.raises(ValueError, match="does not accept parameters"):
            await ExcludedParams.from_seed_group_async(
                seed_group,
                next_message=_make_message("user", "Should fail"),
            )
