# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for AttackIdentifier-specific functionality.

Note: Base Identifier functionality (hash computation, to_dict/from_dict basics,
frozen/hashable properties) is tested via ScorerIdentifier in test_scorer_identifier.py.
These tests focus on AttackIdentifier-specific fields and from_dict deserialization
of nested sub-identifiers.
"""

import pytest

from pyrit.identifiers import AttackIdentifier, ConverterIdentifier, ScorerIdentifier, TargetIdentifier


def _make_target_identifier() -> TargetIdentifier:
    return TargetIdentifier(
        class_name="OpenAIChatTarget",
        class_module="pyrit.prompt_target.openai.openai_chat_target",
        class_description="OpenAI chat target",
        identifier_type="instance",
        endpoint="https://api.openai.com/v1",
        model_name="gpt-4o",
    )


def _make_scorer_identifier() -> ScorerIdentifier:
    return ScorerIdentifier(
        class_name="SelfAskTrueFalseScorer",
        class_module="pyrit.score.true_false.self_ask_true_false_scorer",
        class_description="True/false scorer",
        identifier_type="instance",
    )


def _make_converter_identifier() -> ConverterIdentifier:
    return ConverterIdentifier(
        class_name="Base64Converter",
        class_module="pyrit.prompt_converter.base64_converter",
        class_description="Base64 converter",
        identifier_type="instance",
        supported_input_types=["text"],
        supported_output_types=["text"],
    )


class TestAttackIdentifierCreation:
    """Test basic AttackIdentifier creation."""

    def test_creation_minimal(self):
        """Test creating an AttackIdentifier with only base fields."""
        identifier = AttackIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
        )

        assert identifier.class_name == "PromptSendingAttack"
        assert identifier.objective_target_identifier is None
        assert identifier.objective_scorer_identifier is None
        assert identifier.request_converter_identifiers is None
        assert identifier.attack_specific_params is None
        assert identifier.hash is not None

    def test_creation_all_fields(self):
        """Test creating an AttackIdentifier with all sub-identifiers."""
        target_id = _make_target_identifier()
        scorer_id = _make_scorer_identifier()
        converter_id = _make_converter_identifier()

        identifier = AttackIdentifier(
            class_name="CrescendoAttack",
            class_module="pyrit.executor.attack.multi_turn.crescendo",
            objective_target_identifier=target_id,
            objective_scorer_identifier=scorer_id,
            request_converter_identifiers=[converter_id],
            attack_specific_params={"max_turns": 10},
        )

        assert identifier.objective_target_identifier is target_id
        assert identifier.objective_scorer_identifier is scorer_id
        assert identifier.request_converter_identifiers == [converter_id]
        assert identifier.attack_specific_params == {"max_turns": 10}

    def test_frozen(self):
        """Test that AttackIdentifier is immutable."""
        identifier = AttackIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
        )

        with pytest.raises(AttributeError):
            identifier.class_name = "Other"  # type: ignore[misc]

    def test_hashable(self):
        """Test that AttackIdentifier can be used in sets/dicts."""
        identifier = AttackIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
        )
        # Should not raise
        {identifier}
        {identifier: 1}


class TestAttackIdentifierFromDict:
    """Test AttackIdentifier.from_dict with nested sub-identifier deserialization."""

    def test_from_dict_minimal(self):
        """Test from_dict with no nested sub-identifiers."""
        data = {
            "class_name": "PromptSendingAttack",
            "class_module": "pyrit.executor.attack.single_turn.prompt_sending",
        }

        result = AttackIdentifier.from_dict(data)

        assert isinstance(result, AttackIdentifier)
        assert result.class_name == "PromptSendingAttack"
        assert result.objective_target_identifier is None
        assert result.objective_scorer_identifier is None
        assert result.request_converter_identifiers is None

    def test_from_dict_deserializes_nested_target(self):
        """Test that from_dict recursively deserializes the target sub-identifier."""
        target_id = _make_target_identifier()
        data = {
            "class_name": "PromptSendingAttack",
            "class_module": "pyrit.executor.attack.single_turn.prompt_sending",
            "objective_target_identifier": target_id.to_dict(),
        }

        result = AttackIdentifier.from_dict(data)

        assert isinstance(result.objective_target_identifier, TargetIdentifier)
        assert result.objective_target_identifier.class_name == "OpenAIChatTarget"
        assert result.objective_target_identifier.endpoint == "https://api.openai.com/v1"

    def test_from_dict_deserializes_nested_scorer(self):
        """Test that from_dict recursively deserializes the scorer sub-identifier."""
        scorer_id = _make_scorer_identifier()
        data = {
            "class_name": "CrescendoAttack",
            "class_module": "pyrit.executor.attack.multi_turn.crescendo",
            "objective_scorer_identifier": scorer_id.to_dict(),
        }

        result = AttackIdentifier.from_dict(data)

        assert isinstance(result.objective_scorer_identifier, ScorerIdentifier)
        assert result.objective_scorer_identifier.class_name == "SelfAskTrueFalseScorer"

    def test_from_dict_deserializes_nested_converters(self):
        """Test that from_dict recursively deserializes converter sub-identifiers."""
        converter_id = _make_converter_identifier()
        data = {
            "class_name": "PromptSendingAttack",
            "class_module": "pyrit.executor.attack.single_turn.prompt_sending",
            "request_converter_identifiers": [converter_id.to_dict()],
        }

        result = AttackIdentifier.from_dict(data)

        assert result.request_converter_identifiers is not None
        assert len(result.request_converter_identifiers) == 1
        assert isinstance(result.request_converter_identifiers[0], ConverterIdentifier)
        assert result.request_converter_identifiers[0].class_name == "Base64Converter"

    def test_from_dict_all_nested(self):
        """Test from_dict with all nested sub-identifiers as dicts."""
        target_id = _make_target_identifier()
        scorer_id = _make_scorer_identifier()
        converter_id = _make_converter_identifier()

        data = {
            "class_name": "CrescendoAttack",
            "class_module": "pyrit.executor.attack.multi_turn.crescendo",
            "objective_target_identifier": target_id.to_dict(),
            "objective_scorer_identifier": scorer_id.to_dict(),
            "request_converter_identifiers": [converter_id.to_dict()],
            "attack_specific_params": {"max_turns": 10},
        }

        result = AttackIdentifier.from_dict(data)

        assert isinstance(result, AttackIdentifier)
        assert isinstance(result.objective_target_identifier, TargetIdentifier)
        assert isinstance(result.objective_scorer_identifier, ScorerIdentifier)
        assert isinstance(result.request_converter_identifiers[0], ConverterIdentifier)
        assert result.attack_specific_params == {"max_turns": 10}

    def test_from_dict_already_typed_sub_identifiers_not_re_parsed(self):
        """Test that from_dict handles already-typed sub-identifiers without error."""
        target_id = _make_target_identifier()
        converter_id = _make_converter_identifier()

        data = {
            "class_name": "PromptSendingAttack",
            "class_module": "pyrit.executor.attack.single_turn.prompt_sending",
            "objective_target_identifier": target_id,  # Already typed, not a dict
            "request_converter_identifiers": [converter_id],  # Already typed
        }

        result = AttackIdentifier.from_dict(data)

        assert result.objective_target_identifier is target_id
        assert result.request_converter_identifiers[0] is converter_id

    def test_from_dict_none_converters_stays_none(self):
        """Test that None converter list is preserved as None."""
        data = {
            "class_name": "PromptSendingAttack",
            "class_module": "pyrit.executor.attack.single_turn.prompt_sending",
            "request_converter_identifiers": None,
        }

        result = AttackIdentifier.from_dict(data)
        assert result.request_converter_identifiers is None


class TestAttackIdentifierRoundTrip:
    """Test to_dict → from_dict round-trip fidelity."""

    def test_round_trip_minimal(self):
        """Test round-trip with minimal fields."""
        original = AttackIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
        )

        restored = AttackIdentifier.from_dict(original.to_dict())

        assert restored.class_name == original.class_name
        assert restored.class_module == original.class_module
        assert restored.hash == original.hash

    def test_round_trip_with_nested_identifiers(self):
        """Test round-trip preserves nested sub-identifiers."""
        original = AttackIdentifier(
            class_name="CrescendoAttack",
            class_module="pyrit.executor.attack.multi_turn.crescendo",
            objective_target_identifier=_make_target_identifier(),
            objective_scorer_identifier=_make_scorer_identifier(),
            request_converter_identifiers=[_make_converter_identifier()],
        )

        restored = AttackIdentifier.from_dict(original.to_dict())

        assert isinstance(restored.objective_target_identifier, TargetIdentifier)
        assert isinstance(restored.objective_scorer_identifier, ScorerIdentifier)
        assert isinstance(restored.request_converter_identifiers[0], ConverterIdentifier)
        assert restored.hash == original.hash

    def test_round_trip_with_attack_specific_params(self):
        """Test round-trip preserves attack_specific_params."""
        original = AttackIdentifier(
            class_name="TreeOfAttacks",
            class_module="pyrit.executor.attack.multi_turn.tree_of_attacks",
            attack_specific_params={"width": 3, "depth": 5, "pruning": True},
        )

        restored = AttackIdentifier.from_dict(original.to_dict())

        assert restored.attack_specific_params == {"width": 3, "depth": 5, "pruning": True}
        assert restored.hash == original.hash
