# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
from pathlib import Path
from textwrap import dedent
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.common.path import SCORER_CONFIG_PATH
from pyrit.exceptions import InvalidJsonException, remove_markdown_json
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer


class MockScorer(Scorer):
    async def _score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
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
async def test_scorer_score_value_with_llm_use_provided_attack_identifier(good_json):
    scorer = MockScorer()
    scorer.scorer_type = "true_false"

    prompt_response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json)]
    )
    chat_target = MagicMock(PromptChatTarget)
    chat_target.send_prompt_async = AsyncMock(return_value=prompt_response)
    chat_target.set_system_prompt = MagicMock()

    expected_system_prompt = "system_prompt"
    expected_attack_id = "attack_id"
    expected_scored_prompt_id = "123"

    await scorer._score_value_with_llm(
        prompt_target=chat_target,
        system_prompt=expected_system_prompt,
        prompt_request_value="prompt_request_value",
        prompt_request_data_type="text",
        scored_prompt_id=expected_scored_prompt_id,
        category="category",
        task="task",
        attack_identifier={"id": expected_attack_id},
    )

    chat_target.set_system_prompt.assert_called_once()

    _, set_sys_prompt_args = chat_target.set_system_prompt.call_args
    assert set_sys_prompt_args["system_prompt"] == expected_system_prompt
    assert isinstance(set_sys_prompt_args["conversation_id"], str)
    assert set_sys_prompt_args["attack_identifier"]["id"] == expected_attack_id
    assert set_sys_prompt_args["attack_identifier"]["scored_prompt_id"] == expected_scored_prompt_id


@pytest.mark.asyncio
async def test_scorer_score_value_with_llm_does_not_add_score_prompt_id_for_empty_attack_identifier(good_json):
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
    assert not set_sys_prompt_args["attack_identifier"]


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


def test_scorer_path_verification_rejection():
    """
    Test that the scorer correctly refuses to verify a non-existent path.
    """
    scorer = MockScorer()
    mock_path: str = "this/does/not/exist.yaml"
    with pytest.raises(ValueError, match="Path not found"):
        scorer._verify_and_resolve_path(mock_path)


def test_scorer_path_verification_confirmation():
    """
    Test that the scorer verifies the paths that currently exist
    under the scorer configs.
    """
    scorer = MockScorer()
    all_yamls_as_str: list[str] = []
    full_paths: list[str] = []
    for root, dirs, files in os.walk(SCORER_CONFIG_PATH):
        full_paths.extend([os.path.join(root, f) for f in files if f.endswith(".yaml")])
        all_yamls_as_str.extend([f for f in files if f.endswith(".yaml")])
    resolved_paths = [Path(p).resolve() for p in full_paths]
    attempted_paths = [scorer._verify_and_resolve_path(p) for p in full_paths]
    assert attempted_paths == resolved_paths


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

    # TEMPORARY fix means there should only be 2 scores, one per scorer, and each scorer scores only the first piece
    assert len(result) == 2
    assert score1_1 in result
    assert score2_1 in result
    scorer1.score_async.assert_any_call(request_response=piece1, task="test task")
    scorer2.score_async.assert_any_call(request_response=piece1, task="test task")

    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # Should have 4 scores total (2 scorers x 2 assistant pieces)
    # assert len(result) == 4
    # assert score1_1 in result
    # assert score1_2 in result
    # assert score2_1 in result
    # assert score2_2 in result

    # # Verify each scorer was called twice (once per assistant piece)
    # assert scorer1.score_async.call_count == 2
    # assert scorer2.score_async.call_count == 2

    # # Verify the correct pieces were passed
    # scorer1.score_async.assert_any_call(request_response=piece1, task="test task")
    # scorer1.score_async.assert_any_call(request_response=piece2, task="test task")
    # scorer2.score_async.assert_any_call(request_response=piece1, task="test task")
    # scorer2.score_async.assert_any_call(request_response=piece2, task="test task")


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

    # TEMPORARY fix means each scorer should only be called once for the first piece
    assert scorer1.score_async.call_count == 1
    assert scorer2.score_async.call_count == 1
    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # All scorers should be called for all pieces
    # assert scorer1.score_async.call_count == 2
    # assert scorer2.score_async.call_count == 2


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

    # TEMPORARY fix means there should only be 2 calls, one per PromptRequestResponse
    assert len(call_order) == 2
    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # Verify that for each piece, both scorers are called before moving to next piece
    # # (parallel execution per piece, but sequential piece processing)
    # assert len(call_order) == 4
    # First piece should be scored by both scorers
    assert ("scorer1", "response1") in call_order[:2]
    assert ("scorer2", "response1") in call_order[:2]
    # # Then second piece
    # assert ("scorer1", "response2") in call_order[2:]
    # assert ("scorer2", "response2") in call_order[2:]


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_skip_on_error_true():
    """Test that score_response_select_first_success_async skips error pieces when skip_on_error=True."""
    # Create pieces with mixed error states
    piece1 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock score for successful piece
    score = MagicMock(spec=Score)
    score.get_value.return_value = True

    scorer = MockScorer()
    scorer.score_async = AsyncMock(return_value=[score])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], task="test task", skip_on_error=True
    )

    # Should skip error piece and find success in piece2
    assert result == score
    # Scorer should only be called once (for piece2)
    assert scorer.score_async.call_count == 1
    scorer.score_async.assert_called_with(request_response=piece2, task="test task")


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_skip_on_error_false():
    """Test that score_response_select_first_success_async includes error pieces when skip_on_error=False."""
    # Create pieces with mixed error states
    piece1 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores
    score1 = MagicMock(spec=Score)
    score1.get_value.return_value = True  # Success even from error piece

    score2 = MagicMock(spec=Score)
    score2.get_value.return_value = False

    scorer = MockScorer()
    scorer.score_async = AsyncMock(side_effect=[[score1], [score2]])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], task="test task", skip_on_error=False
    )

    # Should return success from error piece
    assert result == score1
    # Scorer should only be called once (found success in piece1)
    assert scorer.score_async.call_count == 1
    scorer.score_async.assert_called_with(request_response=piece1, task="test task")


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_all_errors_skip_on_error_true():
    """
    Test that score_response_select_first_success_async returns None when all pieces
    have errors and skip_on_error=True.
    """
    # Create pieces that all have errors
    piece1 = PromptRequestPiece(role="assistant", original_value="error1", response_error="blocked")
    piece2 = PromptRequestPiece(role="assistant", original_value="error2", response_error="processing")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    scorer = MockScorer()
    scorer.score_async = AsyncMock(return_value=[MagicMock()])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], task="test task", skip_on_error=True
    )

    # Should return None and not call scorer
    assert result is None
    scorer.score_async.assert_not_called()


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_all_errors_skip_on_error_false():
    """Test that score_response_select_first_success_async processes error pieces when skip_on_error=False."""
    # Create pieces that all have errors
    piece1 = PromptRequestPiece(role="assistant", original_value="error1", response_error="blocked")
    piece2 = PromptRequestPiece(role="assistant", original_value="error2", response_error="processing")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores
    score1 = MagicMock(spec=Score)
    score1.get_value.return_value = False

    score2 = MagicMock(spec=Score)
    score2.get_value.return_value = False

    scorer = MockScorer()
    scorer.score_async = AsyncMock(side_effect=[[score1], [score2]])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], task="test task", skip_on_error=False
    )

    # Should process error pieces and return first score
    assert result == score1

    # TEMPORARY fix means the scorer should only be called once for the first piece
    assert scorer.score_async.call_count == 1
    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # Should have called scorer for both pieces
    # assert scorer.score_async.call_count == 2


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_mixed_errors_skip_on_error_multiple_scorers():
    """Test score_response_select_first_success_async with multiple scorers and mixed error pieces."""
    # Create pieces with mixed error states
    piece1 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")
    piece2 = PromptRequestPiece(role="assistant", original_value="good response")
    piece3 = PromptRequestPiece(role="assistant", original_value="another error", response_error="unknown")

    response = PromptRequestResponse(request_pieces=[piece1, piece2, piece3])

    # Create mock scores
    score1 = MagicMock(spec=Score)
    score1.get_value.return_value = False

    score2 = MagicMock(spec=Score)
    score2.get_value.return_value = True  # Success

    scorer1 = MockScorer()
    scorer1.score_async = AsyncMock(return_value=[score1])

    scorer2 = MockScorer()
    scorer2.score_async = AsyncMock(return_value=[score2])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer1, scorer2], task="test task", skip_on_error=True
    )

    # Should find success in piece2 with scorer2
    assert result == score2
    # Both scorers should be called once (only for piece2)
    assert scorer1.score_async.call_count == 1
    assert scorer2.score_async.call_count == 1
    # Verify they were called with the non-error piece
    scorer1.score_async.assert_called_with(request_response=piece2, task="test task")
    scorer2.score_async.assert_called_with(request_response=piece2, task="test task")


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_skip_on_error_no_success():
    """Test score_response_select_first_success_async returns first score when no success found with skip_on_error."""
    piece1 = PromptRequestPiece(role="assistant", original_value="good response")
    piece2 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")
    piece3 = PromptRequestPiece(role="assistant", original_value="another good response")

    response = PromptRequestResponse(request_pieces=[piece1, piece2, piece3])

    # All scores are failures
    score1 = MagicMock(spec=Score)
    score1.get_value.return_value = False

    score2 = MagicMock(spec=Score)
    score2.get_value.return_value = False

    scorer = MockScorer()
    scorer.score_async = AsyncMock(side_effect=[[score1], [score2]])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], task="test task", skip_on_error=True
    )

    # Should return the first score as failure indicator
    assert result == score1

    # Temporary fix means the scorer should only be called once for the first piece
    assert scorer.score_async.call_count == 1
    scorer.score_async.assert_called_with(request_response=piece1, task="test task")
    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # Should have been called twice (for piece1 and piece3, skipping piece2)
    # assert scorer.score_async.call_count == 2
    # scorer.score_async.assert_any_call(request_response=piece1, task="test task")
    # scorer.score_async.assert_any_call(request_response=piece3, task="test task")


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_skip_on_error_empty_scores():
    """Test score_response_select_first_success_async handles empty score lists with skip_on_error."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Scorer returns empty list
    scorer = MockScorer()
    scorer.score_async = AsyncMock(return_value=[])

    result = await Scorer.score_response_select_first_success_async(
        response=response, scorers=[scorer], task="test task", skip_on_error=True
    )

    # Should return None since no scores were produced
    assert result is None
    # Should only be called for non-error piece
    assert scorer.score_async.call_count == 1
    scorer.score_async.assert_called_with(request_response=piece1, task="test task")


@pytest.mark.asyncio
async def test_score_response_with_objective_async_empty_response():
    """Test score_response_with_objective_async with empty response."""
    response = PromptRequestResponse(request_pieces=[])

    aux_scorer = MockScorer()
    obj_scorer = MockScorer()

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=[aux_scorer], objective_scorers=[obj_scorer], task="test task"
    )

    assert result == {"auxiliary_scores": [], "objective_scores": []}


@pytest.mark.asyncio
async def test_score_response_with_objective_async_no_scorers():
    """Test score_response_with_objective_async with no scorers provided."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value="test")])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=None, objective_scorers=None, task="test task"
    )

    assert result == {"auxiliary_scores": [], "objective_scores": []}


@pytest.mark.asyncio
async def test_score_response_with_objective_async_auxiliary_only():
    """Test score_response_with_objective_async with only auxiliary scorers."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Create mock auxiliary scores
    aux_score1 = MagicMock(spec=Score)
    aux_score2 = MagicMock(spec=Score)

    # Create mock auxiliary scorers
    aux_scorer1 = MockScorer()
    aux_scorer1.score_async = AsyncMock(return_value=[aux_score1])

    aux_scorer2 = MockScorer()
    aux_scorer2.score_async = AsyncMock(return_value=[aux_score2])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=[aux_scorer1, aux_scorer2], objective_scorers=None, task="test task"
    )

    # Should have auxiliary scores but no objective scores
    assert len(result["auxiliary_scores"]) == 2
    assert aux_score1 in result["auxiliary_scores"]
    assert aux_score2 in result["auxiliary_scores"]
    assert result["objective_scores"] == []


@pytest.mark.asyncio
async def test_score_response_with_objective_async_objective_only():
    """Test score_response_with_objective_async with only objective scorers."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Create mock objective score
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True

    # Create mock objective scorer
    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[obj_score])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=None, objective_scorers=[obj_scorer], task="test task"
    )

    # Should have objective score but no auxiliary scores
    assert result["auxiliary_scores"] == []
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score


@pytest.mark.asyncio
async def test_score_response_with_objective_async_both_types():
    """Test score_response_with_objective_async with both auxiliary and objective scorers."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Create mock scores
    aux_score = MagicMock(spec=Score)
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = False  # Not successful

    # Create mock scorers
    aux_scorer = MockScorer()
    aux_scorer.score_async = AsyncMock(return_value=[aux_score])

    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[obj_score])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=[aux_scorer], objective_scorers=[obj_scorer], task="test task"
    )

    # Should have both types of scores
    assert len(result["auxiliary_scores"]) == 1
    assert result["auxiliary_scores"][0] == aux_score
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score


@pytest.mark.asyncio
async def test_score_response_with_objective_async_multiple_pieces():
    """Test score_response_with_objective_async with multiple response pieces."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")
    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores
    aux_scores = [MagicMock(spec=Score) for _ in range(4)]  # 2 pieces x 2 scorers
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True  # Success on first piece

    # Create mock auxiliary scorers
    aux_scorer1 = MockScorer()
    aux_scorer1.score_async = AsyncMock(side_effect=[[aux_scores[0]], [aux_scores[1]]])

    aux_scorer2 = MockScorer()
    aux_scorer2.score_async = AsyncMock(side_effect=[[aux_scores[2]], [aux_scores[3]]])

    # Create mock objective scorer
    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[obj_score])

    result = await Scorer.score_response_with_objective_async(
        response=response,
        auxiliary_scorers=[aux_scorer1, aux_scorer2],
        objective_scorers=[obj_scorer],
        task="test task",
    )

    # TEMPORARY fix means there should only be 2 auxiliary scores, one per PromptRequestResponse
    assert len(result["auxiliary_scores"]) == 2

    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # Should have all auxiliary scores
    # assert len(result["auxiliary_scores"]) == 4
    # for score in aux_scores:
    #     assert score in result["auxiliary_scores"]

    # Should have only one objective score (first success)
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score


@pytest.mark.asyncio
async def test_score_response_with_objective_async_role_filter():
    """Test score_response_with_objective_async with different role filters."""
    pieces = [
        PromptRequestPiece(role="assistant", original_value="assistant response"),
        PromptRequestPiece(role="user", original_value="user input"),
        PromptRequestPiece(role="system", original_value="system message"),
    ]
    response = PromptRequestResponse(request_pieces=pieces)

    # Create mock scores
    aux_score = MagicMock(spec=Score)
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True

    # Create mock scorers with tracking
    aux_scored_pieces = []
    obj_scored_pieces = []

    async def track_aux_score(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        aux_scored_pieces.append(request_response)
        return [aux_score]

    async def track_obj_score(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        obj_scored_pieces.append(request_response)
        return [obj_score]

    aux_scorer = MockScorer()
    aux_scorer.score_async = track_aux_score

    obj_scorer = MockScorer()
    obj_scorer.score_async = track_obj_score

    result = await Scorer.score_response_with_objective_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorers=[obj_scorer],
        role_filter="assistant",
        task="test task",
    )

    # Should only score assistant pieces
    assert len(aux_scored_pieces) == 1
    assert aux_scored_pieces[0].role == "assistant"
    assert len(obj_scored_pieces) == 1
    assert obj_scored_pieces[0].role == "assistant"

    assert len(result["auxiliary_scores"]) == 1
    assert len(result["objective_scores"]) == 1


@pytest.mark.asyncio
async def test_score_response_with_objective_async_skip_on_error_true():
    """Test score_response_with_objective_async skips error pieces when skip_on_error=True."""
    piece1 = PromptRequestPiece(role="assistant", original_value="good response")
    piece2 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")
    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores
    aux_score = MagicMock(spec=Score)
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True

    # Create mock scorers
    aux_scorer = MockScorer()
    aux_scorer.score_async = AsyncMock(return_value=[aux_score])

    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[obj_score])

    result = await Scorer.score_response_with_objective_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorers=[obj_scorer],
        task="test task",
        skip_on_error=True,
    )

    # Should only score the non-error piece
    assert len(result["auxiliary_scores"]) == 1
    assert len(result["objective_scores"]) == 1

    # Verify only non-error piece was scored
    aux_scorer.score_async.assert_called_once()
    obj_scorer.score_async.assert_called_once()


@pytest.mark.asyncio
async def test_score_response_with_objective_async_skip_on_error_false():
    """Test score_response_with_objective_async includes error pieces when skip_on_error=False."""
    piece1 = PromptRequestPiece(role="assistant", original_value="good response")
    piece2 = PromptRequestPiece(role="assistant", original_value="error", response_error="blocked")
    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Create mock scores
    aux_scores = [MagicMock(spec=Score), MagicMock(spec=Score)]
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True

    # Create mock scorers
    aux_scorer = MockScorer()
    aux_scorer.score_async = AsyncMock(side_effect=[[aux_scores[0]], [aux_scores[1]]])

    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[obj_score])

    result = await Scorer.score_response_with_objective_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorers=[obj_scorer],
        task="test task",
        skip_on_error=False,
    )

    # Temporary fix means there should only be 1 auxiliary score (first piece)
    assert len(result["auxiliary_scores"]) == 1
    # The following commented-out lines should be uncommented when the permanent solution is implemented
    # # Should score both pieces for auxiliary
    # assert len(result["auxiliary_scores"]) == 2

    # But only one objective score (first success)
    assert len(result["objective_scores"]) == 1

    # # Verify both pieces were scored for auxiliary
    # assert aux_scorer.score_async.call_count == 2


@pytest.mark.asyncio
async def test_score_response_with_objective_async_objective_failure():
    """Test score_response_with_objective_async when no objective succeeds."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Create mock scores (all failures)
    obj_score1 = MagicMock(spec=Score)
    obj_score1.get_value.return_value = False

    obj_score2 = MagicMock(spec=Score)
    obj_score2.get_value.return_value = False

    # Create mock objective scorers
    obj_scorer1 = MockScorer()
    obj_scorer1.score_async = AsyncMock(return_value=[obj_score1])

    obj_scorer2 = MockScorer()
    obj_scorer2.score_async = AsyncMock(return_value=[obj_score2])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=None, objective_scorers=[obj_scorer1, obj_scorer2], task="test task"
    )

    # Should return the first score as failure indicator
    assert result["auxiliary_scores"] == []
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score1


@pytest.mark.asyncio
async def test_score_response_with_objective_async_concurrent_execution():
    """Test that auxiliary and objective scoring happen concurrently."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Track call order to verify concurrent execution
    call_order = []

    async def mock_aux_score_async(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        call_order.append("aux_start")
        # Simulate some async work
        await asyncio.sleep(0.01)
        call_order.append("aux_end")
        return [MagicMock(spec=Score)]

    async def mock_obj_score_async(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        call_order.append("obj_start")
        # Simulate some async work
        await asyncio.sleep(0.01)
        call_order.append("obj_end")
        score = MagicMock(spec=Score)
        score.get_value.return_value = True
        return [score]

    aux_scorer = MockScorer()
    aux_scorer.score_async = mock_aux_score_async

    obj_scorer = MockScorer()
    obj_scorer.score_async = mock_obj_score_async

    await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=[aux_scorer], objective_scorers=[obj_scorer], task="test task"
    )

    # Both should start before either finishes (concurrent execution)
    assert call_order.index("aux_start") < call_order.index("obj_end")
    assert call_order.index("obj_start") < call_order.index("aux_end")


@pytest.mark.asyncio
async def test_score_response_with_objective_async_empty_lists():
    """Test score_response_with_objective_async with empty scorer lists."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=[], objective_scorers=[], task="test task"
    )

    assert result == {"auxiliary_scores": [], "objective_scores": []}


@pytest.mark.asyncio
async def test_score_response_with_objective_async_no_objective_success():
    """Test score_response_with_objective_async when select_first_success returns None."""
    response = PromptRequestResponse(request_pieces=[])  # Empty response

    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[MagicMock()])

    result = await Scorer.score_response_with_objective_async(
        response=response, auxiliary_scorers=None, objective_scorers=[obj_scorer], task="test task"
    )

    # Should have empty list for objective_scores (not None)
    assert result["auxiliary_scores"] == []
    assert result["objective_scores"] == []
    assert isinstance(result["objective_scores"], list)


@pytest.mark.asyncio
async def test_score_response_with_objective_async_mixed_roles():
    """Test score_response_with_objective_async filters roles correctly."""
    pieces = [
        PromptRequestPiece(role="system", original_value="system prompt"),
        PromptRequestPiece(role="user", original_value="user message"),
        PromptRequestPiece(role="assistant", original_value="assistant response"),
    ]
    response = PromptRequestResponse(request_pieces=pieces)

    # Create mock scores
    aux_score = MagicMock(spec=Score)
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True

    # Create mock scorers with tracking
    aux_scored_pieces = []
    obj_scored_pieces = []

    async def track_aux_score(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        aux_scored_pieces.append(request_response)
        return [aux_score]

    async def track_obj_score(request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        obj_scored_pieces.append(request_response)
        return [obj_score]

    aux_scorer = MockScorer()
    aux_scorer.score_async = track_aux_score

    obj_scorer = MockScorer()
    obj_scorer.score_async = track_obj_score

    result = await Scorer.score_response_with_objective_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorers=[obj_scorer],
        role_filter="assistant",
        task="test task",
    )

    # Should only score assistant pieces
    assert len(aux_scored_pieces) == 1
    assert aux_scored_pieces[0].role == "assistant"
    assert len(obj_scored_pieces) == 1
    assert obj_scored_pieces[0].role == "assistant"

    assert len(result["auxiliary_scores"]) == 1
    assert len(result["objective_scores"]) == 1


def test_get_scorer_metrics(tmp_path):
    from pyrit.score import Scorer
    from pyrit.score.scorer_evaluation.scorer_evaluator import (
        HarmScorerEvaluator,
        HarmScorerMetrics,
    )

    # Create a fake metrics file
    metrics = HarmScorerMetrics(
        mean_absolute_error=0.1,
        mae_standard_error=0.01,
        t_statistic=1.0,
        p_value=0.05,
        krippendorff_alpha_combined=0.8,
        krippendorff_alpha_humans=0.7,
        krippendorff_alpha_model=0.9,
    )
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        f.write(metrics.to_json())
    scorer = MagicMock(spec=Scorer)
    evaluator = HarmScorerEvaluator(scorer)
    # Patch _get_metrics_path to return our temp file
    with patch.object(evaluator, "_get_metrics_path", return_value=metrics_path):
        loaded = evaluator.get_scorer_metrics("any_dataset")
        assert loaded == metrics
