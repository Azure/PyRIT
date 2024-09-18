# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import uuid

from pyrit.models import PromptRequestPiece
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.score import SubStringScorer

from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.mark.asyncio
async def test_score_prompts_by_request_id_async(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_pieces_by_id.return_value = sample_conversations

    scorer = SubStringScorer(
        substring="test",
        category="test",
        memory=memory,
    )
    scorer.score_async = AsyncMock()  # type: ignore

    orchestrator = ScoringOrchestrator(memory=memory)

    await orchestrator.score_prompts_by_request_id_async(scorer=scorer, prompt_ids=["id1"])
    assert scorer.score_async.call_count == len(sample_conversations)


@pytest.mark.asyncio
async def test_score_prompts_by_orchestrator_only_responses(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_piece_by_orchestrator_id.return_value = sample_conversations

    orchestrator = ScoringOrchestrator(memory=memory)
    scorer = MagicMock()

    with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
        await orchestrator.score_prompts_by_orchestrator_id_async(scorer=scorer, orchestrator_ids=[str(uuid.uuid4())])

        mock_score.assert_called_once()
        _, called_kwargs = mock_score.call_args
        for prompt in called_kwargs["prompts"]:
            assert prompt.role == "assistant"


@pytest.mark.asyncio
async def test_score_prompts_by_orchestrator_includes_requests(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_piece_by_orchestrator_id.return_value = sample_conversations

    orchestrator = ScoringOrchestrator(memory=memory)
    scorer = MagicMock()

    with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
        await orchestrator.score_prompts_by_orchestrator_id_async(
            scorer=scorer, orchestrator_ids=[str(uuid.uuid4())], responses_only=False
        )

        mock_score.assert_called_once()
        _, called_kwargs = mock_score.call_args
        roles = [prompt.role for prompt in called_kwargs["prompts"]]
        assert "user" in roles


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_only_responses(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory_labels = {"op_name": "op1", "user_name": "name1"}
    sample_conversations[1].labels = memory_labels
    sample_conversations[2].labels = memory_labels
    memory.get_prompt_request_piece_by_memory_labels.return_value = sample_conversations

    orchestrator = ScoringOrchestrator(memory=memory)
    scorer = MagicMock()

    with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
        await orchestrator.score_prompts_by_memory_labels_async(scorer=scorer, memory_labels=memory_labels)

        mock_score.assert_called_once()
        _, called_kwargs = mock_score.call_args
        for prompt in called_kwargs["prompts"]:
            assert prompt.role == "assistant"
        assert len(called_kwargs["prompts"]) == 2


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_includes_requests(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_piece_by_memory_labels.return_value = sample_conversations
    memory_labels = {"op_name": "op1", "user_name": "name1"}
    orchestrator = ScoringOrchestrator(memory=memory)
    scorer = MagicMock()

    with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
        await orchestrator.score_prompts_by_memory_labels_async(
            scorer=scorer, memory_labels=memory_labels, responses_only=False
        )

        mock_score.assert_called_once()
        _, called_kwargs = mock_score.call_args
        roles = [prompt.role for prompt in called_kwargs["prompts"]]
        assert "user" in roles


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_async_raises_error_empty_memory_labels():
    orchestrator = ScoringOrchestrator(memory=MagicMock())

    with pytest.raises(ValueError, match="Invalid memory_labels: Please provide valid memory labels."):
        await orchestrator.score_prompts_by_memory_labels_async(
            scorer=MagicMock(), memory_labels={}, responses_only=False
        )


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_async_raises_error_no_matching_labels():
    memory = MagicMock()
    memory.get_prompt_request_piece_by_memory_labels.return_value = []
    orchestrator = ScoringOrchestrator(memory=memory)

    with pytest.raises(
        ValueError, match="No entries match the provided memory labels. Please check your memory labels."
    ):
        await orchestrator.score_prompts_by_memory_labels_async(
            scorer=MagicMock(),
            memory_labels={"op_name": "nonexistent_op", "user_name": "nonexistent_user"},
            responses_only=False,
        )
