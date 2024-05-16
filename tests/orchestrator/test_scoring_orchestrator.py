# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator

from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.mark.asyncio
async def test_score_prompts_by_prompt_id_async(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_pieces_by_id.return_value = sample_conversations

    scorer = MagicMock()
    scorer.score_async = AsyncMock()

    orchestrator = ScoringOrchestrator(memory=memory)

    await orchestrator.score_prompts_by_request_id_async(scorer=scorer, prompt_ids=["id1"])
    assert scorer.score_async.call_count == len(sample_conversations)


@pytest.mark.asyncio
async def test_score_prompts_by_orchestrator_only_responses(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_piece_by_orchestrator_id.return_value = sample_conversations

    orchestrator = ScoringOrchestrator(memory=memory)

    with patch.object(orchestrator, "_score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
        await orchestrator.score_prompts_by_orchestrator_id_async(scorer=MagicMock(), orchestrator_ids=[123])

        mock_score.assert_called_once()
        _, called_kwargs = mock_score.call_args
        for prompt in called_kwargs["prompts"]:
            assert prompt.role == "assistant"


@pytest.mark.asyncio
async def test_score_prompts_by_orchestrator_includes_requests(sample_conversations: list[PromptRequestPiece]):

    memory = MagicMock()
    memory.get_prompt_request_piece_by_orchestrator_id.return_value = sample_conversations

    orchestrator = ScoringOrchestrator(memory=memory)

    with patch.object(orchestrator, "_score_prompts_batch_async", new_callable=AsyncMock) as mock_score:
        await orchestrator.score_prompts_by_orchestrator_id_async(
            scorer=MagicMock(), orchestrator_ids=[123], responses_only=False
        )

        mock_score.assert_called_once()
        _, called_kwargs = mock_score.call_args
        roles = [prompt.role for prompt in called_kwargs["prompts"]]
        assert "user" in roles
