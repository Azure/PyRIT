# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib

from pyrit.score.scorer_identifier import ScorerIdentifier


class TestScorerIdentifierBasic:
    """Test basic ScorerIdentifier functionality."""

    def test_scorer_identifier_creation_minimal(self):
        """Test creating a ScorerIdentifier with only required fields."""
        identifier = ScorerIdentifier(type="TestScorer")

        assert identifier.type == "TestScorer"
        assert identifier.system_prompt_template is None
        assert identifier.user_prompt_template is None
        assert identifier.sub_identifier is None
        assert identifier.target_info is None
        assert identifier.score_aggregator is None
        assert identifier.scorer_specific_params is None

    def test_scorer_identifier_creation_all_fields(self):
        """Test creating a ScorerIdentifier with all fields."""
        sub_id = ScorerIdentifier(type="SubScorer")
        identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template="System prompt",
            user_prompt_template="User prompt",
            sub_identifier=[sub_id],
            target_info={"model_name": "gpt-4", "temperature": 0.7},
            score_aggregator="mean",
            scorer_specific_params={"param1": "value1"},
        )

        assert identifier.type == "TestScorer"
        assert identifier.system_prompt_template == "System prompt"
        assert identifier.user_prompt_template == "User prompt"
        assert len(identifier.sub_identifier) == 1
        assert identifier.target_info["model_name"] == "gpt-4"
        assert identifier.score_aggregator == "mean"
        assert identifier.scorer_specific_params["param1"] == "value1"


class TestScorerIdentifierHash:
    """Test hash computation for ScorerIdentifier."""

    def test_compute_hash_deterministic(self):
        """Test that compute_hash returns the same value for identical configurations."""
        identifier1 = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template="Test prompt",
        )
        identifier2 = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template="Test prompt",
        )

        hash1 = identifier1.compute_hash()
        hash2 = identifier2.compute_hash()

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_compute_hash_different_for_different_configs(self):
        """Test that different configurations produce different hashes."""
        identifier1 = ScorerIdentifier(type="TestScorer")
        identifier2 = ScorerIdentifier(type="TestScorer", system_prompt_template="prompt")
        identifier3 = ScorerIdentifier(type="OtherScorer")

        hash1 = identifier1.compute_hash()
        hash2 = identifier2.compute_hash()
        hash3 = identifier3.compute_hash()

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_compute_hash_includes_all_fields(self):
        """Test that hash changes when any field changes."""
        base = ScorerIdentifier(type="TestScorer")
        with_prompt = ScorerIdentifier(type="TestScorer", system_prompt_template="prompt")
        with_target_info = ScorerIdentifier(type="TestScorer", target_info={"model_name": "gpt-4"})
        with_aggregator = ScorerIdentifier(type="TestScorer", score_aggregator="mean")

        base_hash = base.compute_hash()
        prompt_hash = with_prompt.compute_hash()
        target_hash = with_target_info.compute_hash()
        aggregator_hash = with_aggregator.compute_hash()

        # All should be different
        hashes = [base_hash, prompt_hash, target_hash, aggregator_hash]
        assert len(set(hashes)) == 4, "All hashes should be unique"


class TestScorerIdentifierCompactDict:
    """Test to_compact_dict method for ScorerIdentifier."""

    def test_to_compact_dict_basic(self):
        """Test basic to_compact_dict output."""
        identifier = ScorerIdentifier(type="TestScorer")

        result = identifier.to_compact_dict()

        assert result["__type__"] == "TestScorer"
        assert result["system_prompt_template"] is None
        assert result["user_prompt_template"] is None
        assert result["sub_identifier"] is None
        assert result["target_info"] is None
        assert result["score_aggregator"] is None
        assert result["scorer_specific_params"] is None
        assert "hash" in result
        assert isinstance(result["hash"], str)
        assert len(result["hash"]) == 64  # SHA256 hex digest length

    def test_to_compact_dict_uses_type_key(self):
        """Test that __type__ key is used (not 'type')."""
        identifier = ScorerIdentifier(type="TestScorer")

        result = identifier.to_compact_dict()

        assert "__type__" in result
        assert "type" not in result

    def test_to_compact_dict_short_prompt_preserved(self):
        """Test that short prompts (<= 100 chars) are preserved as-is."""
        short_prompt = "A" * 100  # Exactly 100 characters
        identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template=short_prompt,
            user_prompt_template=short_prompt,
        )

        result = identifier.to_compact_dict()

        assert result["system_prompt_template"] == short_prompt
        assert result["user_prompt_template"] == short_prompt

    def test_to_compact_dict_long_prompt_hashed(self):
        """Test that long prompts (> 100 chars) are hashed."""
        long_prompt = "A" * 101  # Just over 100 characters
        expected_hash = hashlib.sha256(long_prompt.encode()).hexdigest()[:16]

        identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template=long_prompt,
            user_prompt_template=long_prompt,
        )

        result = identifier.to_compact_dict()

        assert result["system_prompt_template"] == f"sha256:{expected_hash}"
        assert result["user_prompt_template"] == f"sha256:{expected_hash}"

    def test_to_compact_dict_very_long_prompt_hashed(self):
        """Test that very long prompts are properly hashed."""
        very_long_prompt = "Test prompt with lots of content. " * 100

        identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template=very_long_prompt,
        )

        result = identifier.to_compact_dict()

        assert result["system_prompt_template"].startswith("sha256:")
        assert len(result["system_prompt_template"]) == 23  # "sha256:" + 16 chars

    def test_to_compact_dict_recursive_sub_identifier(self):
        """Test that sub_identifiers are recursively compacted."""
        long_prompt = "X" * 150

        sub_identifier = ScorerIdentifier(
            type="SubScorer",
            system_prompt_template=long_prompt,
        )

        identifier = ScorerIdentifier(
            type="ParentScorer",
            sub_identifier=[sub_identifier],
        )

        result = identifier.to_compact_dict()

        assert result["sub_identifier"] is not None
        assert len(result["sub_identifier"]) == 1
        assert result["sub_identifier"][0]["__type__"] == "SubScorer"
        assert result["sub_identifier"][0]["system_prompt_template"].startswith("sha256:")

    def test_to_compact_dict_multiple_sub_identifiers(self):
        """Test compacting multiple sub_identifiers."""
        sub1 = ScorerIdentifier(type="SubScorer1")
        sub2 = ScorerIdentifier(type="SubScorer2")

        identifier = ScorerIdentifier(
            type="ParentScorer",
            sub_identifier=[sub1, sub2],
        )

        result = identifier.to_compact_dict()

        assert len(result["sub_identifier"]) == 2
        assert result["sub_identifier"][0]["__type__"] == "SubScorer1"
        assert result["sub_identifier"][1]["__type__"] == "SubScorer2"

    def test_to_compact_dict_nested_sub_identifiers(self):
        """Test deeply nested sub_identifiers."""
        innermost = ScorerIdentifier(type="Innermost")
        middle = ScorerIdentifier(type="Middle", sub_identifier=[innermost])
        outer = ScorerIdentifier(type="Outer", sub_identifier=[middle])

        result = outer.to_compact_dict()

        assert result["sub_identifier"][0]["__type__"] == "Middle"
        assert result["sub_identifier"][0]["sub_identifier"][0]["__type__"] == "Innermost"


class TestScorerIdentifierHashConsistency:
    """Test that hash and to_compact_dict are consistent."""

    def test_hash_included_in_compact_dict(self):
        """Test that to_compact_dict includes the computed hash."""
        long_prompt = "B" * 200

        identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template=long_prompt,
        )

        compact_dict = identifier.to_compact_dict()

        # Hash should be included in compact_dict
        assert "hash" in compact_dict
        assert compact_dict["hash"] == identifier.compute_hash()

    def test_hash_consistent_with_stored_format(self):
        """Test that hash computed from a scorer matches hash in stored format."""
        # This simulates what happens when we store and retrieve from registry
        identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template="A" * 150,  # Long prompt that gets hashed
            target_info={"model_name": "gpt-4"},
        )

        # Original hash
        original_hash = identifier.compute_hash()

        # Simulate stored format (what registry stores)
        stored_format = identifier.to_compact_dict()

        # The stored format should include the hash
        assert stored_format["hash"] == original_hash

    def test_hash_stable_across_calls(self):
        """Test that multiple calls to compute_hash return the same value."""
        identifier = ScorerIdentifier(
            type="TestScorer",
            scorer_specific_params={"key": "value"},
        )

        hashes = [identifier.compute_hash() for _ in range(10)]

        assert len(set(hashes)) == 1, "All hash values should be identical"


class TestScorerIdentifierWithSubIdentifiers:
    """Test ScorerIdentifier with sub_identifiers (composite scorers)."""

    def test_sub_identifier_affects_hash(self):
        """Test that sub_identifier differences affect the hash."""
        sub1 = ScorerIdentifier(type="SubScorer")
        sub2 = ScorerIdentifier(type="SubScorer", system_prompt_template="different")

        parent1 = ScorerIdentifier(type="Parent", sub_identifier=[sub1])
        parent2 = ScorerIdentifier(type="Parent", sub_identifier=[sub2])

        assert parent1.compute_hash() != parent2.compute_hash()

    def test_sub_identifier_order_affects_hash(self):
        """Test that sub_identifier order affects the hash."""
        sub_a = ScorerIdentifier(type="ScorerA")
        sub_b = ScorerIdentifier(type="ScorerB")

        parent1 = ScorerIdentifier(type="Parent", sub_identifier=[sub_a, sub_b])
        parent2 = ScorerIdentifier(type="Parent", sub_identifier=[sub_b, sub_a])

        assert parent1.compute_hash() != parent2.compute_hash()


class TestScorerIdentifierScoreAggregator:
    """Test score_aggregator field behavior."""

    def test_score_aggregator_in_compact_dict(self):
        """Test that score_aggregator appears in compact dict."""
        identifier = ScorerIdentifier(
            type="TestScorer",
            score_aggregator="majority_vote",
        )

        result = identifier.to_compact_dict()

        assert result["score_aggregator"] == "majority_vote"

    def test_score_aggregator_affects_hash(self):
        """Test that score_aggregator affects the hash."""
        id1 = ScorerIdentifier(type="TestScorer", score_aggregator="mean")
        id2 = ScorerIdentifier(type="TestScorer", score_aggregator="max")
        id3 = ScorerIdentifier(type="TestScorer")

        hash1 = id1.compute_hash()
        hash2 = id2.compute_hash()
        hash3 = id3.compute_hash()

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3


class TestScorerIdentifierPyritVersion:
    """Test pyrit_version field behavior."""

    def test_pyrit_version_default(self):
        """Test that pyrit_version is set by default."""
        import pyrit

        identifier = ScorerIdentifier(type="TestScorer")

        assert identifier.pyrit_version == pyrit.__version__

    def test_pyrit_version_in_compact_dict(self):
        """Test that pyrit_version appears in compact dict."""
        import pyrit

        identifier = ScorerIdentifier(type="TestScorer")
        result = identifier.to_compact_dict()

        assert result["pyrit_version"] == pyrit.__version__

    def test_pyrit_version_can_be_overridden(self):
        """Test that pyrit_version can be explicitly set."""
        identifier = ScorerIdentifier(
            type="TestScorer",
            pyrit_version="0.0.1-test",
        )

        assert identifier.pyrit_version == "0.0.1-test"
        result = identifier.to_compact_dict()
        assert result["pyrit_version"] == "0.0.1-test"


class TestScorerSubclassIdentifiers:
    """Test that scorer subclasses correctly build their identifiers."""

    def test_true_false_composite_scorer_identifier(self, patch_central_database):
        """Test TrueFalseCompositeScorer builds identifier with sub-scorers and aggregator."""
        from unittest.mock import MagicMock

        from pyrit.score import (
            TrueFalseCompositeScorer,
            TrueFalseScoreAggregator,
            TrueFalseScorer,
        )

        # Create mock sub-scorers
        mock_scorer1 = MagicMock(spec=TrueFalseScorer)
        mock_scorer1.scorer_identifier = ScorerIdentifier(type="MockScorer1")

        mock_scorer2 = MagicMock(spec=TrueFalseScorer)
        mock_scorer2.scorer_identifier = ScorerIdentifier(type="MockScorer2")

        # Create composite scorer
        composite = TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND,
            scorers=[mock_scorer1, mock_scorer2],
        )

        identifier = composite.scorer_identifier

        assert identifier.type == "TrueFalseCompositeScorer"
        assert identifier.score_aggregator == "AND_"
        assert identifier.sub_identifier is not None
        assert len(identifier.sub_identifier) == 2
        assert identifier.sub_identifier[0].type == "MockScorer1"
        assert identifier.sub_identifier[1].type == "MockScorer2"
        # No system prompt or target for composite scorer
        assert identifier.system_prompt_template is None
        assert identifier.target_info is None

    def test_float_scale_threshold_scorer_identifier(self, patch_central_database):
        """Test FloatScaleThresholdScorer builds identifier with sub-scorer and threshold."""
        from unittest.mock import MagicMock, PropertyMock

        from pyrit.score import FloatScaleScoreAggregator, FloatScaleThresholdScorer

        # Create mock float scale scorer
        mock_scorer = MagicMock()
        mock_scorer.scorer_identifier = ScorerIdentifier(type="MockFloatScorer")
        type(mock_scorer).scorer_identifier = PropertyMock(return_value=ScorerIdentifier(type="MockFloatScorer"))

        threshold_scorer = FloatScaleThresholdScorer(
            scorer=mock_scorer,
            threshold=0.7,
            float_scale_aggregator=FloatScaleScoreAggregator.MAX,
        )

        identifier = threshold_scorer.scorer_identifier

        assert identifier.type == "FloatScaleThresholdScorer"
        assert identifier.sub_identifier is not None
        assert len(identifier.sub_identifier) == 1
        assert identifier.sub_identifier[0].type == "MockFloatScorer"
        assert identifier.scorer_specific_params is not None
        assert identifier.scorer_specific_params["threshold"] == 0.7
        assert identifier.scorer_specific_params["float_scale_aggregator"] == "MAX_"

    def test_self_ask_true_false_scorer_identifier(self, patch_central_database):
        """Test SelfAskTrueFalseScorer builds identifier with system prompt and target."""
        from unittest.mock import MagicMock

        from pyrit.prompt_target import PromptChatTarget
        from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestion

        # Create mock chat target
        mock_target = MagicMock(spec=PromptChatTarget)
        mock_target.get_identifier.return_value = {
            "__type__": "MockChatTarget",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "top_p": 0.9,
        }

        scorer = SelfAskTrueFalseScorer(
            chat_target=mock_target,
            true_false_question=TrueFalseQuestion(
                true_description="Response contains harmful content",
                false_description="Response is safe",
                category="harm",
            ),
        )

        identifier = scorer.scorer_identifier

        assert identifier.type == "SelfAskTrueFalseScorer"
        assert identifier.system_prompt_template is not None
        assert "harmful content" in identifier.system_prompt_template
        assert identifier.target_info is not None
        assert identifier.target_info["__type__"] == "MockChatTarget"
        assert identifier.target_info["model_name"] == "gpt-4"
        assert identifier.target_info["temperature"] == 0.7
        assert identifier.score_aggregator == "OR_"  # Default aggregator

    def test_scorer_identifier_hash_changes_with_threshold(self, patch_central_database):
        """Test that changing threshold produces different hash."""
        from unittest.mock import MagicMock, PropertyMock

        from pyrit.score import FloatScaleThresholdScorer

        mock_scorer = MagicMock()
        type(mock_scorer).scorer_identifier = PropertyMock(return_value=ScorerIdentifier(type="MockFloatScorer"))

        scorer1 = FloatScaleThresholdScorer(scorer=mock_scorer, threshold=0.5)
        scorer2 = FloatScaleThresholdScorer(scorer=mock_scorer, threshold=0.7)

        hash1 = scorer1.scorer_identifier.compute_hash()
        hash2 = scorer2.scorer_identifier.compute_hash()

        assert hash1 != hash2

    def test_scorer_identifier_hash_changes_with_aggregator(self, patch_central_database):
        """Test that changing aggregator produces different hash."""
        from unittest.mock import MagicMock

        from pyrit.score import (
            TrueFalseCompositeScorer,
            TrueFalseScoreAggregator,
            TrueFalseScorer,
        )

        mock_scorer = MagicMock(spec=TrueFalseScorer)
        mock_scorer.scorer_identifier = ScorerIdentifier(type="MockScorer")

        composite_and = TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND,
            scorers=[mock_scorer],
        )
        composite_or = TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.OR,
            scorers=[mock_scorer],
        )

        hash_and = composite_and.scorer_identifier.compute_hash()
        hash_or = composite_or.scorer_identifier.compute_hash()

        assert hash_and != hash_or

    def test_scorer_identifier_hash_changes_with_sub_scorers(self, patch_central_database):
        """Test that changing sub-scorers produces different hash."""
        from unittest.mock import MagicMock

        from pyrit.score import (
            TrueFalseCompositeScorer,
            TrueFalseScoreAggregator,
            TrueFalseScorer,
        )

        mock_scorer1 = MagicMock(spec=TrueFalseScorer)
        mock_scorer1.scorer_identifier = ScorerIdentifier(type="MockScorer1")

        mock_scorer2 = MagicMock(spec=TrueFalseScorer)
        mock_scorer2.scorer_identifier = ScorerIdentifier(type="MockScorer2")

        composite1 = TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND,
            scorers=[mock_scorer1],
        )
        composite2 = TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND,
            scorers=[mock_scorer1, mock_scorer2],
        )

        hash1 = composite1.scorer_identifier.compute_hash()
        hash2 = composite2.scorer_identifier.compute_hash()

        assert hash1 != hash2

    def test_scorer_identifier_hash_changes_with_target_model(self, patch_central_database):
        """Test that changing target model produces different hash."""
        from unittest.mock import MagicMock

        from pyrit.prompt_target import PromptChatTarget
        from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestion

        question = TrueFalseQuestion(
            true_description="Test true",
            false_description="Test false",
            category="test",
        )

        mock_target1 = MagicMock(spec=PromptChatTarget)
        mock_target1.get_identifier.return_value = {
            "__type__": "MockChatTarget",
            "model_name": "gpt-4",
            "temperature": 0.7,
        }

        mock_target2 = MagicMock(spec=PromptChatTarget)
        mock_target2.get_identifier.return_value = {
            "__type__": "MockChatTarget",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
        }

        scorer1 = SelfAskTrueFalseScorer(chat_target=mock_target1, true_false_question=question)
        scorer2 = SelfAskTrueFalseScorer(chat_target=mock_target2, true_false_question=question)

        hash1 = scorer1.scorer_identifier.compute_hash()
        hash2 = scorer2.scorer_identifier.compute_hash()

        assert hash1 != hash2

    def test_scorer_identifier_same_config_same_hash(self, patch_central_database):
        """Test that identical configurations produce the same hash."""
        from unittest.mock import MagicMock, PropertyMock

        from pyrit.score import FloatScaleThresholdScorer

        mock_scorer = MagicMock()
        type(mock_scorer).scorer_identifier = PropertyMock(return_value=ScorerIdentifier(type="MockFloatScorer"))

        scorer1 = FloatScaleThresholdScorer(scorer=mock_scorer, threshold=0.5)
        scorer2 = FloatScaleThresholdScorer(scorer=mock_scorer, threshold=0.5)

        hash1 = scorer1.scorer_identifier.compute_hash()
        hash2 = scorer2.scorer_identifier.compute_hash()

        assert hash1 == hash2

    def test_scorer_get_identifier_returns_compact_dict(self, patch_central_database):
        """Test that scorer.get_identifier() returns compact dict with hash."""
        from unittest.mock import MagicMock

        from pyrit.score import (
            TrueFalseCompositeScorer,
            TrueFalseScoreAggregator,
            TrueFalseScorer,
        )

        mock_scorer = MagicMock(spec=TrueFalseScorer)
        mock_scorer.scorer_identifier = ScorerIdentifier(type="MockScorer")

        composite = TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND,
            scorers=[mock_scorer],
        )

        identifier_dict = composite.get_identifier()

        assert "__type__" in identifier_dict
        assert identifier_dict["__type__"] == "TrueFalseCompositeScorer"
        assert "hash" in identifier_dict
        assert len(identifier_dict["hash"]) == 64  # SHA256 hex digest
        assert "score_aggregator" in identifier_dict
        assert identifier_dict["score_aggregator"] == "AND_"
