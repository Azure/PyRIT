# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the SeedGroup and SeedAttackGroup classes."""

import uuid

import pytest

from pyrit.models.seeds import (
    SeedAttackGroup,
    SeedGroup,
    SeedObjective,
    SeedPrompt,
    SeedSimulatedConversation,
)


# =============================================================================
# SeedGroup Tests
# =============================================================================


class TestSeedGroupInit:
    """Tests for SeedGroup initialization."""

    def test_init_with_single_prompt(self):
        """Test initialization with a single SeedPrompt."""
        prompt = SeedPrompt(value="Test prompt", data_type="text")
        group = SeedGroup(seeds=[prompt])

        assert len(group.seeds) == 1
        assert group.prompts[0].value == "Test prompt"

    def test_init_with_multiple_prompts(self):
        """Test initialization with multiple SeedPrompts."""
        prompts = [
            SeedPrompt(value="Prompt 1", data_type="text", sequence=0, role="user"),
            SeedPrompt(value="Prompt 2", data_type="text", sequence=1, role="assistant"),
        ]
        group = SeedGroup(seeds=prompts)

        assert len(group.seeds) == 2
        assert len(group.prompts) == 2

    def test_init_with_objective_and_prompts(self):
        """Test initialization with objective and prompts."""
        objective = SeedObjective(value="Test objective")
        prompt = SeedPrompt(value="Test prompt", data_type="text")
        group = SeedGroup(seeds=[objective, prompt])

        assert len(group.seeds) == 2
        # Objective should be first
        assert isinstance(group.seeds[0], SeedObjective)

    def test_init_with_dict_seeds(self):
        """Test initialization with dictionary seeds."""
        group = SeedGroup(
            seeds=[
                {"value": "Test objective", "is_objective": True},
                {"value": "Test prompt", "data_type": "text"},
            ]
        )

        assert len(group.seeds) == 2
        assert isinstance(group.seeds[0], SeedObjective)
        assert isinstance(group.seeds[1], SeedPrompt)

    def test_init_empty_raises_error(self):
        """Test that empty seeds raises ValueError."""
        with pytest.raises(ValueError, match="SeedGroup cannot be empty"):
            SeedGroup(seeds=[])

    def test_init_multiple_objectives_raises_error(self):
        """Test that multiple objectives raises ValueError."""
        with pytest.raises(ValueError, match="SeedGroups can only have one objective"):
            SeedGroup(
                seeds=[
                    SeedObjective(value="Objective 1"),
                    SeedObjective(value="Objective 2"),
                ]
            )

    def test_init_assigns_consistent_group_id(self):
        """Test that all seeds get the same prompt_group_id."""
        prompts = [
            SeedPrompt(value="Prompt 1", data_type="text"),
            SeedPrompt(value="Prompt 2", data_type="text"),
        ]
        group = SeedGroup(seeds=prompts)

        group_ids = {seed.prompt_group_id for seed in group.seeds}
        assert len(group_ids) == 1
        assert None not in group_ids

    def test_init_preserves_existing_group_id(self):
        """Test that existing group_id is preserved."""
        existing_id = uuid.uuid4()
        prompts = [
            SeedPrompt(value="Prompt 1", data_type="text", prompt_group_id=existing_id),
            SeedPrompt(value="Prompt 2", data_type="text"),
        ]
        group = SeedGroup(seeds=prompts)

        for seed in group.seeds:
            assert seed.prompt_group_id == existing_id

    def test_init_sorts_prompts_by_sequence(self):
        """Test that prompts are sorted by sequence."""
        prompts = [
            SeedPrompt(value="Prompt 2", data_type="text", sequence=2, role="user"),
            SeedPrompt(value="Prompt 0", data_type="text", sequence=0, role="user"),
            SeedPrompt(value="Prompt 1", data_type="text", sequence=1, role="assistant"),
        ]
        group = SeedGroup(seeds=prompts)

        # Check prompts are in order
        assert group.prompts[0].value == "Prompt 0"
        assert group.prompts[1].value == "Prompt 1"
        assert group.prompts[2].value == "Prompt 2"

    def test_init_objective_first_then_sorted_prompts(self):
        """Test that objective comes first, then sorted prompts."""
        seeds = [
            SeedPrompt(value="Prompt 2", data_type="text", sequence=2, role="user"),
            SeedObjective(value="Objective"),
            SeedPrompt(value="Prompt 0", data_type="text", sequence=0, role="assistant"),
        ]
        group = SeedGroup(seeds=seeds)

        assert isinstance(group.seeds[0], SeedObjective)
        assert group.seeds[1].value == "Prompt 0"
        assert group.seeds[2].value == "Prompt 2"


class TestSeedGroupHarmCategories:
    """Tests for SeedGroup.harm_categories property."""

    def test_harm_categories_empty(self):
        """Test harm_categories with no categories."""
        prompt = SeedPrompt(value="Test", data_type="text")
        group = SeedGroup(seeds=[prompt])

        assert group.harm_categories == []

    def test_harm_categories_from_single_seed(self):
        """Test harm_categories from a single seed."""
        prompt = SeedPrompt(
            value="Test",
            data_type="text",
            harm_categories=["violence", "hate"],
        )
        group = SeedGroup(seeds=[prompt])

        assert set(group.harm_categories) == {"violence", "hate"}

    def test_harm_categories_deduplicated(self):
        """Test that harm_categories are deduplicated."""
        prompts = [
            SeedPrompt(value="Test 1", data_type="text", harm_categories=["violence"]),
            SeedPrompt(value="Test 2", data_type="text", harm_categories=["violence", "hate"]),
        ]
        group = SeedGroup(seeds=prompts)

        assert set(group.harm_categories) == {"violence", "hate"}


# =============================================================================
# SeedAttackGroup Tests
# =============================================================================


class TestSeedAttackGroupInit:
    """Tests for SeedAttackGroup initialization."""

    def test_init_with_objective_and_prompt(self):
        """Test basic initialization with objective and prompt."""
        group = SeedAttackGroup(
            seeds=[
                SeedObjective(value="Test objective"),
                SeedPrompt(value="Test prompt", data_type="text"),
            ]
        )

        assert group.objective is not None
        assert group.objective.value == "Test objective"
        assert len(group.prompts) == 1

    def test_init_with_simulated_conversation(self):
        """Test initialization with simulated conversation config."""
        group = SeedAttackGroup(
            seeds=[
                SeedObjective(value="Test objective"),
                SeedSimulatedConversation(
                    value="",
                    num_turns=3,
                    adversarial_system_prompt="Adversarial",
                    simulated_target_system_prompt="Target",
                ),
            ]
        )

        assert group.has_simulated_conversation
        assert group.simulated_conversation_config is not None
        assert group.simulated_conversation_config.num_turns == 3

    def test_init_with_dict_simulated_conversation(self):
        """Test initialization with dict-based simulated conversation."""
        group = SeedAttackGroup(
            seeds=[
                {"value": "Test objective", "is_objective": True},
                {
                    "is_simulated_conversation": True,
                    "num_turns": 5,
                    "adversarial_system_prompt": "Adversarial",
                },
            ]
        )

        assert group.has_simulated_conversation
        assert group.simulated_conversation_config.num_turns == 5

    def test_init_simulated_conversation_with_multi_sequence_raises_error(self):
        """Test that simulated_conversation with multi-sequence prompts raises error."""
        with pytest.raises(ValueError, match="Cannot use simulated_conversation with multi-sequence prompts"):
            SeedAttackGroup(
                seeds=[
                    SeedObjective(value="Objective"),
                    SeedSimulatedConversation(
                        value="",
                        num_turns=3,
                        adversarial_system_prompt="Adversarial",
                    ),
                    SeedPrompt(value="Prompt 1", data_type="text", sequence=0, role="user"),
                    SeedPrompt(value="Prompt 2", data_type="text", sequence=1, role="assistant"),
                ]
            )

    def test_init_ordering_objective_simulated_prompts(self):
        """Test that seeds are ordered: objective, simulated_conversation, prompts."""
        group = SeedAttackGroup(
            seeds=[
                {"value": "Prompt", "data_type": "text"},
                {"is_simulated_conversation": True, "num_turns": 2, "adversarial_system_prompt": "adv"},
                {"value": "Objective", "is_objective": True},
            ]
        )

        assert isinstance(group.seeds[0], SeedObjective)
        assert isinstance(group.seeds[1], SeedSimulatedConversation)
        assert isinstance(group.seeds[2], SeedPrompt)


class TestSeedAttackGroupObjective:
    """Tests for SeedAttackGroup objective handling."""

    def test_objective_property_returns_objective(self):
        """Test that objective property returns the SeedObjective."""
        group = SeedAttackGroup(
            seeds=[
                SeedObjective(value="My objective"),
                SeedPrompt(value="Test", data_type="text"),
            ]
        )

        assert group.objective.value == "My objective"

    def test_objective_property_none_when_no_objective(self):
        """Test that objective property returns None when no objective."""
        group = SeedAttackGroup(
            seeds=[SeedPrompt(value="Test", data_type="text")]
        )

        assert group.objective is None

    def test_set_objective_updates_existing(self):
        """Test that set_objective updates existing objective."""
        group = SeedAttackGroup(
            seeds=[
                SeedObjective(value="Old objective"),
                SeedPrompt(value="Test", data_type="text"),
            ]
        )

        group.set_objective("New objective")

        assert group.objective.value == "New objective"

    def test_set_objective_creates_new_when_none(self):
        """Test that set_objective creates new objective when none exists."""
        group = SeedAttackGroup(
            seeds=[SeedPrompt(value="Test", data_type="text")]
        )

        group.set_objective("New objective")

        assert group.objective is not None
        assert group.objective.value == "New objective"


class TestSeedAttackGroupSimulatedConversation:
    """Tests for SeedAttackGroup simulated conversation handling."""

    def test_has_simulated_conversation_false_when_none(self):
        """Test has_simulated_conversation is False when no config."""
        group = SeedAttackGroup(
            seeds=[SeedPrompt(value="Test", data_type="text")]
        )

        assert not group.has_simulated_conversation

    def test_has_simulated_conversation_true_when_present(self):
        """Test has_simulated_conversation is True when config present."""
        group = SeedAttackGroup(
            seeds=[
                SeedSimulatedConversation(
                    value="",
                    num_turns=3,
                    adversarial_system_prompt="Adversarial",
                ),
            ]
        )

        assert group.has_simulated_conversation

    def test_simulated_conversation_generated_false_initially(self):
        """Test simulated_conversation_generated is False initially."""
        group = SeedAttackGroup(
            seeds=[
                SeedSimulatedConversation(
                    value="",
                    num_turns=3,
                    adversarial_system_prompt="Adversarial",
                ),
            ]
        )

        assert not group.simulated_conversation_generated


class TestSeedAttackGroupMessageExtraction:
    """Tests for SeedAttackGroup message extraction methods."""

    def test_is_single_turn_true_for_single_sequence_no_objective(self):
        """Test is_single_turn is True for single sequence without objective."""
        group = SeedAttackGroup(
            seeds=[SeedPrompt(value="Test", data_type="text")]
        )

        assert group.is_single_turn()

    def test_is_single_turn_false_with_objective(self):
        """Test is_single_turn is False when objective present."""
        group = SeedAttackGroup(
            seeds=[
                SeedObjective(value="Objective"),
                SeedPrompt(value="Test", data_type="text"),
            ]
        )

        assert not group.is_single_turn()

    def test_is_single_request_true_for_single_sequence(self):
        """Test is_single_request is True for single sequence."""
        group = SeedAttackGroup(
            seeds=[
                SeedPrompt(value="Test 1", data_type="text", sequence=0, role="user"),
                SeedPrompt(value="Test 2", data_type="text", sequence=0, role="user"),
            ]
        )

        assert group.is_single_request()

    def test_is_single_request_false_for_multi_sequence(self):
        """Test is_single_request is False for multi-sequence."""
        group = SeedAttackGroup(
            seeds=[
                SeedPrompt(value="Test 1", data_type="text", sequence=0, role="user"),
                SeedPrompt(value="Test 2", data_type="text", sequence=1, role="assistant"),
            ]
        )

        assert not group.is_single_request()

    def test_next_message_returns_last_user_message(self):
        """Test next_message returns the last user message."""
        group = SeedAttackGroup(
            seeds=[SeedPrompt(value="Test prompt", data_type="text", role="user")]
        )

        next_msg = group.next_message
        assert next_msg is not None
        assert next_msg.get_value() == "Test prompt"

    def test_next_message_none_for_assistant_last(self):
        """Test next_message is None when last message is assistant."""
        group = SeedAttackGroup(
            seeds=[
                SeedPrompt(value="User msg", data_type="text", sequence=0, role="user"),
                SeedPrompt(value="Assistant msg", data_type="text", sequence=1, role="assistant"),
            ]
        )

        assert group.next_message is None

    def test_prepended_conversation_returns_all_except_last_user(self):
        """Test prepended_conversation returns all except last user message."""
        group = SeedAttackGroup(
            seeds=[
                SeedPrompt(value="User 1", data_type="text", sequence=0, role="user"),
                SeedPrompt(value="Assistant 1", data_type="text", sequence=1, role="assistant"),
                SeedPrompt(value="User 2", data_type="text", sequence=2, role="user"),
            ]
        )

        prepended = group.prepended_conversation
        assert prepended is not None
        assert len(prepended) == 2

    def test_user_messages_returns_all_prompts_as_messages(self):
        """Test user_messages returns all prompts as Messages."""
        group = SeedAttackGroup(
            seeds=[
                SeedPrompt(value="Prompt 1", data_type="text", sequence=0, role="user"),
                SeedPrompt(value="Prompt 2", data_type="text", sequence=1, role="assistant"),
            ]
        )

        messages = group.user_messages
        assert len(messages) == 2


class TestSeedAttackGroupRepr:
    """Tests for SeedAttackGroup.__repr__ method."""

    def test_repr_basic(self):
        """Test basic __repr__ output."""
        group = SeedAttackGroup(
            seeds=[SeedPrompt(value="Test", data_type="text")]
        )

        repr_str = repr(group)
        assert "SeedAttackGroup" in repr_str
        assert "seeds=" in repr_str

    def test_repr_with_simulated_conversation(self):
        """Test __repr__ includes simulated indicator."""
        group = SeedAttackGroup(
            seeds=[
                SeedSimulatedConversation(
                    value="",
                    num_turns=3,
                    adversarial_system_prompt="Adversarial",
                ),
            ]
        )

        repr_str = repr(group)
        assert "simulated" in repr_str
