# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid

from pyrit.memory.memory_models import ScoreEntry
from pyrit.models import Score


def test_normalize_scorer_identifier_simple():
    """Test normalizing a simple scorer identifier without sub_identifier."""
    identifier = {
        "__type__": "SelfAskScorer",
        "__module__": "pyrit.score",
        "sub_identifier": None,
    }

    result = ScoreEntry._normalize_scorer_identifier(identifier)

    assert result["__type__"] == "SelfAskScorer"
    assert result["__module__"] == "pyrit.score"
    assert result["sub_identifier"] is None


def test_normalize_scorer_identifier_with_dict():
    """Test normalizing a scorer identifier with a dict sub_identifier."""
    identifier = {
        "__type__": "FloatScaleThresholdScorer",
        "__module__": "pyrit.score",
        "sub_identifier": {
            "__type__": "SelfAskScorer",
            "__module__": "pyrit.score",
            "sub_identifier": None,
        },
    }

    result = ScoreEntry._normalize_scorer_identifier(identifier)

    assert result["__type__"] == "FloatScaleThresholdScorer"
    assert isinstance(result["sub_identifier"], str)

    # Verify it's valid JSON and preserves structure
    parsed = json.loads(result["sub_identifier"])
    assert parsed["__type__"] == "SelfAskScorer"


def test_normalize_scorer_identifier_with_list():
    """Test normalizing a scorer identifier with a list of sub_identifiers."""
    identifier = {
        "__type__": "TrueFalseCompositeScorer",
        "__module__": "pyrit.score",
        "sub_identifier": [
            {"__type__": "ScorerA", "__module__": "pyrit.score", "sub_identifier": None},
            {"__type__": "ScorerB", "__module__": "pyrit.score", "sub_identifier": None},
        ],
    }

    result = ScoreEntry._normalize_scorer_identifier(identifier)

    assert isinstance(result["sub_identifier"], str)

    # Verify it's valid JSON and preserves list
    parsed = json.loads(result["sub_identifier"])
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_denormalize_scorer_identifier_with_dict():
    """Test denormalizing a scorer identifier with a JSON string sub_identifier."""
    sub_id_dict = {
        "__type__": "SelfAskScorer",
        "__module__": "pyrit.score",
        "sub_identifier": None,
    }

    identifier = {
        "__type__": "FloatScaleThresholdScorer",
        "__module__": "pyrit.score",
        "sub_identifier": json.dumps(sub_id_dict),
    }

    result = ScoreEntry._denormalize_scorer_identifier(identifier)

    assert isinstance(result["sub_identifier"], dict)
    assert result["sub_identifier"]["__type__"] == "SelfAskScorer"


def test_score_entry_roundtrip():
    """Test that a Score can be stored and retrieved with proper identifier handling."""
    # Create a score with a nested identifier
    scorer_identifier = {
        "__type__": "FloatScaleThresholdScorer",
        "__module__": "pyrit.score",
        "sub_identifier": {
            "__type__": "SelfAskScorer",
            "__module__": "pyrit.score",
            "sub_identifier": None,
        },
    }

    message_piece_id = uuid.uuid4()
    original_score = Score(
        score_value="0.85",
        score_value_description="High confidence",
        score_type="float_scale",
        score_category=["test"],
        score_rationale="Test rationale",
        score_metadata={"key": "value"},
        scorer_class_identifier=scorer_identifier,
        message_piece_id=message_piece_id,
        objective="test objective",
    )

    # Create ScoreEntry (normalizes on init)
    score_entry = ScoreEntry(entry=original_score)

    # Verify normalization happened
    assert isinstance(score_entry.scorer_class_identifier["sub_identifier"], str)

    # Get score back (denormalizes)
    retrieved_score = score_entry.get_score()

    # Verify denormalization restored the structure
    assert isinstance(retrieved_score.scorer_class_identifier["sub_identifier"], dict)
    assert retrieved_score.scorer_class_identifier["sub_identifier"]["__type__"] == "SelfAskScorer"  # type: ignore
    assert retrieved_score.scorer_class_identifier["__type__"] == "FloatScaleThresholdScorer"


def test_score_entry_roundtrip_with_list():
    """Test roundtrip with a list of sub_identifiers (composite scorer)."""
    scorer_identifier = {
        "__type__": "TrueFalseCompositeScorer",
        "__module__": "pyrit.score",
        "sub_identifier": [
            {"__type__": "ScorerA", "__module__": "pyrit.score", "sub_identifier": None},
            {"__type__": "ScorerB", "__module__": "pyrit.score", "sub_identifier": None},
        ],
    }

    message_piece_id = uuid.uuid4()
    original_score = Score(
        score_value="True",
        score_value_description="Composite result",
        score_type="true_false",
        score_category=["test"],
        score_rationale="Test rationale",
        score_metadata={"key": "value"},
        scorer_class_identifier=scorer_identifier,
        message_piece_id=message_piece_id,
        objective="test objective",
    )

    # Create ScoreEntry (normalizes on init)
    score_entry = ScoreEntry(entry=original_score)

    # Verify normalization happened
    assert isinstance(score_entry.scorer_class_identifier["sub_identifier"], str)

    # Get score back (denormalizes)
    retrieved_score = score_entry.get_score()

    # Verify denormalization restored the list structure
    assert isinstance(retrieved_score.scorer_class_identifier["sub_identifier"], list)
    assert len(retrieved_score.scorer_class_identifier["sub_identifier"]) == 2
    assert retrieved_score.scorer_class_identifier["sub_identifier"][0]["__type__"] == "ScorerA"  # type: ignore
