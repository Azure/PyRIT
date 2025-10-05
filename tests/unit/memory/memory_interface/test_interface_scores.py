# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import uuid
from typing import Literal, Sequence
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.memory import MemoryInterface, PromptMemoryEntry
from pyrit.models import (
    PromptRequestPiece,
    Score,
    SeedPrompt,
)


def test_get_scores_by_attack_id_and_label(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):
    # create list of scores that are associated with sample conversation entries
    # assert that that list of scores is the same as expected :-)

    prompt_id = sample_conversations[0].id

    sqlite_instance.add_request_pieces_to_memory(request_pieces=sample_conversations)

    score = Score(
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )

    sqlite_instance.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = sqlite_instance.get_prompt_scores(attack_id=sample_conversations[0].attack_identifier["id"])

    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].prompt_request_response_id == score.prompt_request_response_id

    db_score = sqlite_instance.get_prompt_scores(labels=sample_conversations[0].labels)
    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value

    db_score = sqlite_instance.get_scores(score_ids=[str(score.id)])
    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value

    db_score = sqlite_instance.get_prompt_scores(
        attack_id=sample_conversations[0].attack_identifier["id"],
        labels={"x": "y"},
    )
    assert len(db_score) == 0

    db_score = sqlite_instance.get_prompt_scores(
        attack_id=str(uuid.uuid4()),
    )
    assert len(db_score) == 0

    db_score = sqlite_instance.get_scores()
    assert len(db_score) == 0


@pytest.mark.parametrize("score_type", ["float_scale", "true_false"])
def test_add_score_get_score(
    sqlite_instance: MemoryInterface,
    sample_conversation_entries: Sequence[PromptMemoryEntry],
    score_type: Literal["float_scale"] | Literal["true_false"],
):
    prompt_id = sample_conversation_entries[0].id

    sqlite_instance._insert_entries(entries=sample_conversation_entries)

    score_value = str(True) if score_type == "true_false" else "0.8"

    score = Score(
        score_value=score_value,
        score_value_description="High score",
        score_type=score_type,
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )

    sqlite_instance.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = sqlite_instance.get_prompt_scores(prompt_ids=[prompt_id])
    assert db_score
    assert len(db_score) == 1
    assert db_score[0].score_value == score_value
    assert db_score[0].score_value_description == "High score"
    assert db_score[0].score_type == score_type
    assert db_score[0].score_category == "test"
    assert db_score[0].score_rationale == "Test score"
    assert db_score[0].score_metadata == "Test metadata"
    assert db_score[0].scorer_class_identifier == {"__type__": "TestScorer"}
    assert db_score[0].prompt_request_response_id == prompt_id


def test_add_score_duplicate_prompt(sqlite_instance: MemoryInterface):
    # Ensure that scores of duplicate prompts are linked back to the original
    original_id = uuid4()
    attack = PromptSendingAttack(objective_target=MagicMock())
    conversation_id = str(uuid4())
    pieces = [
        PromptRequestPiece(
            id=original_id,
            role="assistant",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            attack_identifier=attack.get_identifier(),
        )
    ]
    new_attack_id = str(uuid4())
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    sqlite_instance.duplicate_conversation(new_attack_id=new_attack_id, conversation_id=conversation_id)
    dupe_piece = sqlite_instance.get_prompt_request_pieces(attack_id=new_attack_id)[0]
    dupe_id = dupe_piece.id

    score_id = uuid4()
    # score with prompt_request_response_id as dupe_id
    score = Score(
        id=score_id,
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=dupe_id,
    )
    sqlite_instance.add_scores_to_memory(scores=[score])

    assert score.prompt_request_response_id == original_id
    assert sqlite_instance.get_prompt_scores(prompt_ids=[str(dupe_id)])[0].id == score_id
    assert sqlite_instance.get_prompt_scores(prompt_ids=[str(original_id)])[0].id == score_id


def test_get_scores_by_memory_labels(sqlite_instance: MemoryInterface):
    prompt_id = uuid4()
    pieces = [
        PromptRequestPiece(
            id=prompt_id,
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            sequence=0,
            labels={"sample": "label"},
        )
    ]
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)

    score = Score(
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )
    sqlite_instance.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = sqlite_instance.get_prompt_scores(labels={"sample": "label"})

    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].prompt_request_response_id == prompt_id


@pytest.mark.asyncio
async def test_get_seed_prompts_no_filters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts()
    assert len(result) == 2
    assert result[0].value == "prompt1"
    assert result[1].value == "prompt2"
