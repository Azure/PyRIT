import pytest
from unittest.mock import AsyncMock, MagicMock
from pyrit.score.look_back_scorer import LookBackScorer
from pyrit.models import PromptRequestPiece


@pytest.mark.asyncio
async def test_score_async_success():
    # Arrange
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = [
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="User message", role="user", original_prompt_id="123", orchestrator_identifier="test"
                )
            ]
        )
    ]
    mock_prompt_target = MagicMock()
    mock_unvalidated_score = MagicMock(raw_score_value=0.8)
    mock_unvalidated_score.to_score.return_value = MagicMock(
        score_value=0.8, score_value_description="High", score_rationale="Valid rationale"
    )

    scorer = LookBackScorer(chat_target=mock_prompt_target)
    scorer._memory = mock_memory
    scorer._score_value_with_llm = AsyncMock(return_value=mock_unvalidated_score)

    request_piece = PromptRequestPiece(conversation_id="test_conversation")

    # Act
    scores = await scorer.score_async(request_piece)

    # Assert
    assert len(scores) == 1
    assert scores[0].score_value == 0.8
    assert scores[0].score_value_description == "High"
    assert scores[0].score_rationale == "Valid rationale"
    mock_memory.get_conversation.assert_called_once_with(conversation_id="test_conversation")
    scorer._score_value_with_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_score_async_conversation_not_found():
    # Arrange
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = None
    mock_prompt_target = MagicMock()

    scorer = LookBackScorer(chat_target=mock_prompt_target)
    scorer._memory = mock_memory

    request_piece = PromptRequestPiece(conversation_id="nonexistent_conversation")

    # Act & Assert
    with pytest.raises(ValueError, match="Conversation with ID nonexistent_conversation not found in memory."):
        await scorer.score_async(request_piece)

    mock_memory.get_conversation.assert_called_once_with(conversation_id="nonexistent_conversation")
