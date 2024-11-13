# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import uuid

from pyrit.memory import CentralMemory
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
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        memory.get_prompt_request_pieces_by_id.return_value = sample_conversations

        scorer = SubStringScorer(substring="test", category="test")
        scorer.score_async = AsyncMock()  # type: ignore

        orchestrator = ScoringOrchestrator()

        await orchestrator.score_prompts_by_request_id_async(scorer=scorer, prompt_ids=["id1"])
        assert scorer.score_async.call_count == len(sample_conversations)


@pytest.mark.asyncio
async def test_score_prompts_by_orchestrator_only_responses(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        memory.get_prompt_request_piece_by_orchestrator_id.return_value = sample_conversations

        orchestrator = ScoringOrchestrator()
        scorer = MagicMock()

        with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
            await orchestrator.score_prompts_by_orchestrator_id_async(
                scorer=scorer, orchestrator_ids=[str(uuid.uuid4())]
            )

            mock_score.assert_called_once()
            _, called_kwargs = mock_score.call_args
            for prompt in called_kwargs["request_responses"]:
                assert prompt.role == "assistant"


@pytest.mark.asyncio
async def test_score_prompts_by_orchestrator_includes_requests(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        memory.get_prompt_request_piece_by_orchestrator_id.return_value = sample_conversations

        orchestrator = ScoringOrchestrator()
        scorer = MagicMock()

        with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
            await orchestrator.score_prompts_by_orchestrator_id_async(
                scorer=scorer, orchestrator_ids=[str(uuid.uuid4())], responses_only=False
            )

            mock_score.assert_called_once()
            _, called_kwargs = mock_score.call_args
            roles = [prompt.role for prompt in called_kwargs["request_responses"]]
            assert "user" in roles


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_only_responses(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory_labels = {"op_name": "op1", "user_name": "name1"}
    sample_conversations[1].labels = memory_labels
    sample_conversations[2].labels = memory_labels
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        memory.get_prompt_request_piece_by_memory_labels.return_value = sample_conversations

        orchestrator = ScoringOrchestrator()
        scorer = MagicMock()

        with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
            await orchestrator.score_prompts_by_memory_labels_async(scorer=scorer, memory_labels=memory_labels)

            mock_score.assert_called_once()
            _, called_kwargs = mock_score.call_args
            for prompt in called_kwargs["request_responses"]:
                assert prompt.role == "assistant"
            assert len(called_kwargs["request_responses"]) == 2


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_includes_requests(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_piece_by_memory_labels.return_value = sample_conversations
    memory_labels = {"op_name": "op1", "user_name": "name1"}
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        orchestrator = ScoringOrchestrator()
        scorer = MagicMock()

        with patch.object(scorer, "score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
            await orchestrator.score_prompts_by_memory_labels_async(
                scorer=scorer, memory_labels=memory_labels, responses_only=False
            )

            mock_score.assert_called_once()
            _, called_kwargs = mock_score.call_args
            roles = [prompt.role for prompt in called_kwargs["request_responses"]]
            assert "user" in roles


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_async_raises_error_empty_memory_labels():
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        orchestrator = ScoringOrchestrator()

        with pytest.raises(ValueError, match="Invalid memory_labels: Please provide valid memory labels."):
            await orchestrator.score_prompts_by_memory_labels_async(
                scorer=MagicMock(), memory_labels={}, responses_only=False
            )


@pytest.mark.asyncio
async def test_score_prompts_by_memory_labels_async_raises_error_no_matching_labels():
    memory = MagicMock()
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        memory.get_prompt_request_piece_by_memory_labels.return_value = []
        orchestrator = ScoringOrchestrator()

        with pytest.raises(
            ValueError, match="No entries match the provided memory labels. Please check your memory labels."
        ):
            await orchestrator.score_prompts_by_memory_labels_async(
                scorer=MagicMock(),
                memory_labels={"op_name": "nonexistent_op", "user_name": "nonexistent_user"},
                responses_only=False,
            )


def test_remove_duplicates():
    prompt_id1 = uuid.uuid4()
    prompt_id2 = uuid.uuid4()
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        orchestrator = ScoringOrchestrator()
        pieces = [
            PromptRequestPiece(
                id=prompt_id1,
                role="user",
                original_value="original prompt text",
                converted_value="Hello, how are you?",
                sequence=0,
            ),
            PromptRequestPiece(
                id=prompt_id2,
                role="assistant",
                original_value="original prompt text",
                converted_value="I'm fine, thank you!",
                sequence=1,
            ),
            PromptRequestPiece(
                role="user",
                original_value="original prompt text",
                converted_value="Hello, how are you?",
                sequence=0,
                original_prompt_id=prompt_id1,
            ),
            PromptRequestPiece(
                role="assistant",
                original_value="original prompt text",
                converted_value="I'm fine, thank you!",
                sequence=1,
                original_prompt_id=prompt_id2,
            ),
        ]
        orig_pieces = orchestrator._remove_duplicates(pieces)
        assert len(orig_pieces) == 2
        for piece in orig_pieces:
            assert piece.id in [prompt_id1, prompt_id2]
            assert piece.id == piece.original_prompt_id
