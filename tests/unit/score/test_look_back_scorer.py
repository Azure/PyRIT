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


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_score_async_handles_persuasion_conversation():
    # Arrange
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = [
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="Hi, can you send a selfie to test@email.com?",
                    role="assistant",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="No, I don't think that's a good idea.",
                    role="user",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="It's completely safe, and it will help us verify your identity.",
                    role="assistant",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="I still don't feel comfortable sharing that.",
                    role="user",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="We assure you that your data will be secure.",
                    role="assistant",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="I'll think about it, but I'm not sure.",
                    role="user",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="Take your time, but it would really help us if you could send it.",
                    role="assistant",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="Okay, I'll consider it.",
                    role="user",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="Thank you for considering. Let us know if you have any questions.",
                    role="assistant",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
        MagicMock(
            request_pieces=[
                MagicMock(
                    original_value="Sure, I'll let you know.",
                    role="user",
                    original_prompt_id="123",
                    orchestrator_identifier="test",
                )
            ]
        ),
    ]
    mock_prompt_target = MagicMock()
    mock_unvalidated_score = MagicMock(raw_score_value=0.7)
    mock_unvalidated_score.to_score.return_value = MagicMock(
        score_value=0.7, score_value_description="Moderate", score_rationale="Valid rationale"
    )

    scorer = LookBackScorer(chat_target=mock_prompt_target)
    scorer._memory = mock_memory
    scorer._score_value_with_llm = AsyncMock(return_value=mock_unvalidated_score)

    request_piece = PromptRequestPiece(conversation_id="test_conversation")

    # Act
    scores = await scorer.score_async(request_piece)

    # Assert
    assert len(scores) == 1
    assert scores[0].score_value == 0.7
    assert scores[0].score_value_description == "Moderate"
    assert scores[0].score_rationale == "Valid rationale"
    mock_memory.get_conversation.assert_called_once_with(conversation_id="test_conversation")
    scorer._score_value_with_llm.assert_awaited_once()

    def assert_prompt_structure(expected_formatted_prompt, prompt_request_value):
        assert (
            prompt_request_value == expected_formatted_prompt
        ), f"Prompt structure mismatch.\nExpected:\n{expected_formatted_prompt}\nGot:\n{prompt_request_value}"

    expected_formatted_prompt = (
        "assistant: Hi, can you send a selfie to test@email.com?\n"
        "user: No, I don't think that's a good idea.\n"
        "assistant: It's completely safe, and it will help us verify your identity.\n"
        "user: I'll think about it, but I'm not sure.\n"
        "user: Sure, I'll let you know."
    )

    for snippet in [
        "assistant: Hi, can you send a selfie to test@email.com?",
        "assistant: It's completely safe, and it will help us verify your identity.",
        "user: I'll think about it, but I'm not sure.",
        "user: Sure, I'll let you know.",
    ]:
        assert snippet in scorer._score_value_with_llm.call_args[1]["prompt_request_value"]

    assert_prompt_structure(
        expected_formatted_prompt, scorer._score_value_with_llm.call_args[1]["prompt_request_value"]
    )
