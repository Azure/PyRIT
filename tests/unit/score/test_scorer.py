# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from textwrap import dedent
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions import InvalidJsonException, remove_markdown_json
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer


class MockScorer(Scorer):
    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        return []

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass


@pytest.fixture
def good_json():
    return (
        dedent(
            """
            {"score_value": "1",
            "rationale": "The message does not contain any code or instructions that could be used to cause harm"}
            """
        )
        .strip()
        .replace("\n", " ")
    )


BAD_JSON = "this is not json"

KEY_ERROR_JSON = (
    dedent(
        """
        {"poor_value": "1",
        "rationale": "The message does not contain any code or instructions that could be used to cause harm"}
        """
    )
    .strip()
    .replace("\n", " ")
)

KEY_ERROR2_JSON = (
    dedent(
        """
        {"score_value": "1",
        "bad_rationale": "The message does not contain any code or instructions that could be used to cause harm"}
        """
    )
    .strip()
    .replace("\n", " ")
)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_json", [BAD_JSON, KEY_ERROR_JSON, KEY_ERROR2_JSON])
async def test_scorer_send_chat_target_async_bad_json_exception_retries(bad_json: str):

    chat_target = MagicMock(PromptChatTarget)

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=bad_json)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    with pytest.raises(InvalidJsonException):
        await scorer._score_value_with_llm(
            prompt_target=chat_target,
            system_prompt="system_prompt",
            prompt_request_value="prompt_request_value",
            prompt_request_data_type="text",
            scored_prompt_id="123",
            category="category",
            task="task",
        )

        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_scorer_score_value_with_llm_exception_display_prompt_id():
    chat_target = MagicMock(PromptChatTarget)
    chat_target.send_prompt_async = AsyncMock(side_effect=Exception("Test exception"))

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    with pytest.raises(Exception, match="Error scoring prompt with original prompt ID: 123"):
        await scorer._score_value_with_llm(
            prompt_target=chat_target,
            system_prompt="system_prompt",
            prompt_request_value="prompt_request_value",
            prompt_request_data_type="text",
            scored_prompt_id="123",
            category="category",
            task="task",
        )


@pytest.mark.asyncio
async def test_scorer_score_value_with_llm_use_provided_orchestrator_identifier(good_json):
    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    prompt_response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target = MagicMock(PromptChatTarget)
    chat_target.send_prompt_async = AsyncMock(return_value=prompt_response)
    chat_target.set_system_prompt = MagicMock()

    expected_system_prompt = "system_prompt"
    expected_orchestrator_id = "orchestrator_id"
    expected_scored_prompt_id = "123"

    await scorer._score_value_with_llm(
        prompt_target=chat_target,
        system_prompt=expected_system_prompt,
        prompt_request_value="prompt_request_value",
        prompt_request_data_type="text",
        scored_prompt_id=expected_scored_prompt_id,
        category="category",
        task="task",
        orchestrator_identifier={"id": expected_orchestrator_id},
    )

    chat_target.set_system_prompt.assert_called_once()

    _, set_sys_prompt_args = chat_target.set_system_prompt.call_args
    assert set_sys_prompt_args["system_prompt"] == expected_system_prompt
    assert isinstance(set_sys_prompt_args["conversation_id"], str)
    assert set_sys_prompt_args["orchestrator_identifier"]["id"] == expected_orchestrator_id
    assert set_sys_prompt_args["orchestrator_identifier"]["scored_prompt_id"] == expected_scored_prompt_id


@pytest.mark.asyncio
async def test_scorer_score_value_with_llm_does_not_add_score_prompt_id_for_empty_orchestrator_identifier(good_json):
    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    prompt_response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target = MagicMock(PromptChatTarget)
    chat_target.send_prompt_async = AsyncMock(return_value=prompt_response)
    chat_target.set_system_prompt = MagicMock()

    expected_system_prompt = "system_prompt"

    await scorer._score_value_with_llm(
        prompt_target=chat_target,
        system_prompt=expected_system_prompt,
        prompt_request_value="prompt_request_value",
        prompt_request_data_type="text",
        scored_prompt_id="123",
        category="category",
        task="task",
    )

    chat_target.set_system_prompt.assert_called_once()

    _, set_sys_prompt_args = chat_target.set_system_prompt.call_args
    assert set_sys_prompt_args["system_prompt"] == expected_system_prompt
    assert isinstance(set_sys_prompt_args["conversation_id"], str)
    assert not set_sys_prompt_args["orchestrator_identifier"]


@pytest.mark.asyncio
async def test_scorer_send_chat_target_async_good_response(good_json):

    chat_target = MagicMock(PromptChatTarget)

    good_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=good_json_resp)

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    await scorer._score_value_with_llm(
        prompt_target=chat_target,
        system_prompt="system_prompt",
        prompt_request_value="prompt_request_value",
        prompt_request_data_type="text",
        scored_prompt_id="123",
        category="category",
        task="task",
    )

    assert chat_target.send_prompt_async.call_count == int(1)


@pytest.mark.asyncio
async def test_scorer_remove_markdown_json_called(good_json):

    chat_target = MagicMock(PromptChatTarget)
    good_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=good_json_resp)

    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    with patch("pyrit.score.scorer.remove_markdown_json", wraps=remove_markdown_json) as mock_remove_markdown_json:
        await scorer._score_value_with_llm(
            prompt_target=chat_target,
            system_prompt="system_prompt",
            prompt_request_value="prompt_request_value",
            prompt_request_data_type="text",
            scored_prompt_id="123",
            category="category",
            task="task",
        )

        mock_remove_markdown_json.assert_called_once()


def test_scorer_extract_task_from_response():
    """
    Test that _extract_task_from_response properly gathers text from the
    last turn. We'll mock out the memory's get_prompt_request_pieces method.
    """
    scorer = MockScorer()
    mock_memory = MagicMock()

    response_piece = PromptRequestPiece(original_value="og prompt", role="assistant", conversation_id="xyz", sequence=2)

    mock_memory.get_prompt_request_pieces.return_value = [
        PromptRequestPiece(role="user", original_value="Not applicable", original_value_data_type="text", sequence=0),
        PromptRequestPiece(
            role="user",
            original_value="User's question about the universe",
            converted_value="Not the task",
            original_value_data_type="text",
            sequence=1,
        ),
        response_piece,
    ]

    with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):

        extracted_task = scorer._extract_task_from_response(response_piece)
        assert "User's question about the universe" in extracted_task


@pytest.mark.asyncio
async def test_scorer_score_responses_batch_async(patch_central_database):
    """
    Test that score_responses_batch_async filters to only assistant pieces,
    calls score_prompts_with_tasks_batch_async, and returns results.
    """
    scorer = MockScorer()

    with patch.object(scorer, "score_prompts_with_tasks_batch_async", new_callable=AsyncMock) as mock_batch:
        fake_scores = [MagicMock(), MagicMock()]
        mock_batch.return_value = fake_scores

        user_piece = PromptRequestPiece(role="user", original_value="Hello user", sequence=1)
        assistant_piece = PromptRequestPiece(role="assistant", original_value="Hello from assistant", sequence=2)

        results = await scorer.score_responses_inferring_tasks_batch_async(
            request_responses=[user_piece, assistant_piece], batch_size=10
        )

        mock_batch.assert_awaited_once()
        _, call_kwargs = mock_batch.call_args

        assert "request_responses" in call_kwargs
        assert "tasks" in call_kwargs
        assert len(call_kwargs["request_responses"]) == 1
        assert call_kwargs["request_responses"][0] == assistant_piece

        assert len(call_kwargs["tasks"]) == 1
        assert results == fake_scores


@pytest.mark.asyncio
async def test_score_response_async_empty_scorers():
    """Test that score_response_async returns empty list when no scorers provided."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value="test")])

    result = await Scorer.score_response_async(response=response, scorers=[], task="test task")

    assert result == []


@pytest.mark.asyncio
async def test_score_response_async_no_matching_role():
    """Test that score_response_async returns empty list when no pieces match role filter."""
    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value="test1"),
            PromptRequestPiece(role="system", original_value="test2"),
        ]
    )

    scorer = MockScorer()
    scorer.score_async = AsyncMock(return_value=[MagicMock()])

    result = await Scorer.score_response_async(
        response=response, scorers=[scorer], role_filter="assistant", task="test task"
    )

    assert result == []
    scorer.score_async.assert_not_called()


@pytest.mark.asyncio
async def test_score_response_async_parallel_execution():
    """Test that score_response_async runs all scorers in parallel on all filtered pieces."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")
    piece3 = PromptRequestPiece(role="user", original_value="user input")

    response = PromptRequestResponse(request_pieces=[piece1, piece2, piece3])

    # Create mock scores
    score1_1 = MagicMock(spec=Score)
    score1_2 = MagicMock(spec=Score)
    score2_1 = MagicMock(spec=Score)
    score2_2 = MagicMock(spec=Score)

    # Create mock scorers
    scorer1 = MockScorer()
    scorer1.score_async = AsyncMock(side_effect=[[score1_1], [score1_2]])

    scorer2 = MockScorer()
    scorer2.score_async = AsyncMock(side_effect=[[score2_1], [score2_2]])

    result = await Scorer.score_response_async(
        response=response, scorers=[scorer1, scorer2], role_filter="assistant", task="test task"
    )

    # Should have 4 scores total (2 scorers x 2 assistant pieces)
    assert len(result) == 4
    assert score1_1 in result
    assert score1_2 in result
    assert score2_1 in result
    assert score2_2 in result

    # Verify each scorer was called twice (once per assistant piece)
    assert scorer1.score_async.call_count == 2
    assert scorer2.score_async.call_count == 2

    # Verify the correct pieces were passed
    scorer1.score_async.assert_any_call(request_response=piece1, task="test task")
    scorer1.score_async.assert_any_call(request_response=piece2, task="test task")
    scorer2.score_async.assert_any_call(request_response=piece1, task="test task")
    scorer2.score_async.assert_any_call(request_response=piece2, task="test task")


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_empty_scorers():
    """Test that score_response_select_first_success_async returns None when no scorers provided."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value="test")])

    result = await Scorer.score_response_select_first_success_async(response=response, scorers=[], task="test task")

    assert result is None


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_no_matching_role():
    """Test that score_response_select_first_success_async returns None when no pieces match role filter."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="user", original_value="test")])

    scorer = MockScorer()
    scorer.score_async = AsyncMock(return_value=[MagicMock()])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], role_filter="assistant", task="test task"
    )

    assert result is None
    scorer.score_async.assert_not_called()


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_finds_success():
    """Test that score_response_select_first_success_async returns first successful score."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores
    score1 = MagicMock(spec=Score)
    score1.get_value.return_value = False  # Failure

    score2 = MagicMock(spec=Score)
    score2.get_value.return_value = True  # Success

    score3 = MagicMock(spec=Score)
    score3.get_value.return_value = True  # Another success (should not be reached)

    # Create mock scorers
    scorer1 = MockScorer()
    scorer1.score_async = AsyncMock(side_effect=[[score1], [score3]])

    scorer2 = MockScorer()
    scorer2.score_async = AsyncMock(return_value=[score2])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer1, scorer2], task="test task"
    )

    # Should return the first successful score (score2)
    assert result == score2

    # scorer1 should be called only once (for piece1)
    assert scorer1.score_async.call_count == 1
    # scorer2 should be called only once (for piece1, returning success)
    assert scorer2.score_async.call_count == 1


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_no_success_returns_first():
    """Test that score_response_select_first_success_async returns first score when no success found."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores (all failures)
    score1 = MagicMock(spec=Score)
    score1.get_value.return_value = False

    score2 = MagicMock(spec=Score)
    score2.get_value.return_value = False

    score3 = MagicMock(spec=Score)
    score3.get_value.return_value = False

    score4 = MagicMock(spec=Score)
    score4.get_value.return_value = False

    # Create mock scorers
    scorer1 = MockScorer()
    scorer1.score_async = AsyncMock(side_effect=[[score1], [score3]])

    scorer2 = MockScorer()
    scorer2.score_async = AsyncMock(side_effect=[[score2], [score4]])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer1, scorer2], task="test task"
    )

    # Should return the first score encountered (score1)
    assert result == score1

    # All scorers should be called for all pieces
    assert scorer1.score_async.call_count == 2
    assert scorer2.score_async.call_count == 2


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_parallel_scoring_per_piece():
    """Test that score_response_select_first_success_async runs scorers in parallel for each piece."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Track call order
    call_order = []

    async def mock_score_async_1(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        call_order.append(("scorer1", request_response.original_value))
        score = MagicMock(spec=Score)
        score.get_value.return_value = False
        return [score]

    async def mock_score_async_2(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        call_order.append(("scorer2", request_response.original_value))
        score = MagicMock(spec=Score)
        score.get_value.return_value = False
        return [score]

    scorer1 = MockScorer()
    scorer1.score_async = mock_score_async_1

    scorer2 = MockScorer()
    scorer2.score_async = mock_score_async_2

    await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer1, scorer2], task="test task"
    )

    # Verify that for each piece, both scorers are called before moving to next piece
    # (parallel execution per piece, but sequential piece processing)
    assert len(call_order) == 4
    # First piece should be scored by both scorers
    assert ("scorer1", "response1") in call_order[:2]
    assert ("scorer2", "response1") in call_order[:2]
    # Then second piece
    assert ("scorer1", "response2") in call_order[2:]
    assert ("scorer2", "response2") in call_order[2:]
