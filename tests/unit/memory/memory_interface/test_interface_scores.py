# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Literal, Sequence
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.memory import MemoryInterface, PromptMemoryEntry
from pyrit.models import (
    MessagePiece,
    Score,
    SeedPrompt,
)


def test_get_scores_by_attack_id_and_label(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[MessagePiece]
):
    # create list of scores that are associated with sample conversation entries
    # assert that that list of scores is the same as expected :-)

    prompt_id = sample_conversations[0].id
    assert prompt_id is not None, "Prompt ID should not be None"

    sqlite_instance.add_message_pieces_to_memory(message_pieces=sample_conversations)

    score = Score(
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category=["test"],
        score_rationale="Test score",
        score_metadata={"test": "metadata"},
        scorer_class_identifier={"__type__": "TestScorer"},
        message_piece_id=prompt_id,
    )

    sqlite_instance.add_scores_to_memory(scores=[score])

    # Test get_scores with score_ids filter
    db_score = sqlite_instance.get_scores(score_ids=[str(score.id)])
    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].message_piece_id == score.message_piece_id

    # Test get_message_pieces returns scores attached to pieces
    pieces = sqlite_instance.get_message_pieces(prompt_ids=[prompt_id])
    assert len(pieces) == 1
    assert len(pieces[0].scores) == 1
    assert pieces[0].scores[0].score_value == score.score_value

    # Test get_scores with no filters returns empty
    db_score = sqlite_instance.get_scores()
    assert len(db_score) == 0


@pytest.mark.parametrize("score_type", ["float_scale", "true_false"])
def test_add_score_get_score(
    sqlite_instance: MemoryInterface,
    sample_conversation_entries: Sequence[PromptMemoryEntry],
    score_type: Literal["float_scale"] | Literal["true_false"],
):
    prompt_id = sample_conversation_entries[0].id
    assert prompt_id is not None, "Prompt ID should not be None"

    sqlite_instance._insert_entries(entries=sample_conversation_entries)

    score_value = str(True) if score_type == "true_false" else "0.8"

    score = Score(
        score_value=score_value,
        score_value_description="High score",
        score_type=score_type,
        score_category=["test"],
        score_rationale="Test score",
        score_metadata={"test": "metadata"},
        scorer_class_identifier={"__type__": "TestScorer"},
        message_piece_id=prompt_id,
    )

    sqlite_instance.add_scores_to_memory(scores=[score])

    # Fetch the score via get_message_pieces which joins scores
    pieces = sqlite_instance.get_message_pieces(prompt_ids=[prompt_id])
    assert pieces
    assert len(pieces) == 1
    db_score = pieces[0].scores
    assert db_score
    assert len(db_score) == 1
    assert db_score[0].score_value == score_value
    assert db_score[0].score_value_description == "High score"
    assert db_score[0].score_type == score_type
    assert db_score[0].score_category == ["test"]
    assert db_score[0].score_rationale == "Test score"
    assert db_score[0].score_metadata == {"test": "metadata"}
    assert db_score[0].scorer_class_identifier == {"__type__": "TestScorer"}
    assert db_score[0].message_piece_id == prompt_id


def test_add_score_duplicate_prompt(sqlite_instance: MemoryInterface):
    # Ensure that scores of duplicate prompts are linked back to the original
    original_id = uuid4()
    attack = PromptSendingAttack(objective_target=MagicMock())
    conversation_id = str(uuid4())
    pieces = [
        MessagePiece(
            id=original_id,
            role="assistant",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            attack_identifier=attack.get_identifier(),
        )
    ]
    sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)
    sqlite_instance.duplicate_conversation(conversation_id=conversation_id)
    # Get the duplicated piece (it will have a different conversation_id but same attack_id)
    all_pieces = sqlite_instance.get_message_pieces()
    dupe_piece = [p for p in all_pieces if p.id != original_id][0]
    dupe_id = dupe_piece.id
    assert dupe_id is not None, "Dupe ID should not be None"

    score_id = uuid4()
    # score with message_piece_id as dupe_id
    score = Score(
        id=score_id,
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category=["test"],
        score_rationale="Test score",
        score_metadata={"test": "metadata"},
        scorer_class_identifier={"__type__": "TestScorer"},
        message_piece_id=dupe_id,
    )
    sqlite_instance.add_scores_to_memory(scores=[score])

    # Score should be linked to original_id
    assert score.message_piece_id == original_id

    # Both dupe and original should retrieve the same score via get_message_pieces
    dupe_pieces = sqlite_instance.get_message_pieces(prompt_ids=[dupe_id])
    assert len(dupe_pieces) == 1
    assert len(dupe_pieces[0].scores) == 1
    assert dupe_pieces[0].scores[0].id == score_id

    original_pieces = sqlite_instance.get_message_pieces(prompt_ids=[original_id])
    assert len(original_pieces) == 1
    assert len(original_pieces[0].scores) == 1
    assert original_pieces[0].scores[0].id == score_id


def test_get_scores_by_memory_labels(sqlite_instance: MemoryInterface):
    prompt_id = uuid4()
    pieces = [
        MessagePiece(
            id=prompt_id,
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            sequence=0,
            labels={"sample": "label"},
        )
    ]
    sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

    score = Score(
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category=["test"],
        score_rationale="Test score",
        score_metadata={"test": "metadata"},
        scorer_class_identifier={"__type__": "TestScorer"},
        message_piece_id=prompt_id,
    )
    sqlite_instance.add_scores_to_memory(scores=[score])

    # Fetch pieces by label and check scores are attached
    pieces_with_label = sqlite_instance.get_message_pieces(labels={"sample": "label"})
    assert len(pieces_with_label) == 1
    db_score = pieces_with_label[0].scores
    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].message_piece_id == prompt_id


@pytest.mark.asyncio
async def test_get_seeds_no_filters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    await sqlite_instance.add_seeds_to_memory_async(seeds=seed_prompts, added_by="test")

    result = sqlite_instance.get_seeds()
    assert len(result) == 2
    assert result[0].value == "prompt1"
    assert result[1].value == "prompt2"


# ===========================================================================================
# DEPRECATED METHOD TESTS - Remove in 0.13.0
# These tests verify deprecated methods still exist and emit warnings.
# Do not add new functionality tests here - use the new methods above instead.
# ===========================================================================================


def test_get_prompt_scores_deprecated_exists(sqlite_instance: MemoryInterface):
    """Verify get_prompt_scores exists and emits deprecation warning. Remove in 0.13.0."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call with no matching data - just verify it exists and warns
        result = sqlite_instance.get_prompt_scores(prompt_ids=["00000000-0000-0000-0000-000000000000"])
        assert len(result) == 0
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "get_prompt_scores is deprecated" in str(w[0].message)
