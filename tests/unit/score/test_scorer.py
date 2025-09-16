# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
from textwrap import dedent
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions import InvalidJsonException, remove_markdown_json
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer, ScorerPromptValidator



@pytest.fixture
def good_json():
    return '{"score_value": "1", "score_rationale": "Valid response"}'

# Add missing BAD_JSON, KEY_ERROR_JSON, KEY_ERROR2_JSON definitions for test parametrize
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



class DummyValidator(ScorerPromptValidator):
    def validate(self, request_response, objective=None):
        pass
    def is_request_piece_supported(self, request_piece):
        return True

class MockScorer(Scorer):
    def __init__(self):
        super().__init__(validator=DummyValidator())
    async def _score_async(self, request_response: PromptRequestResponse, *, objective: Optional[str] = None) -> list[Score]:
        return [Score(score_value="true", score_value_description="desc", score_type="true_false", score_category=None, score_metadata=None, score_rationale="rationale", scorer_class_identifier=self.get_identifier(), prompt_request_response_id="mock_id", objective=objective)]
    async def _score_piece_async(self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None) -> list[Score]:
        return [Score(score_value="true", score_value_description="desc", score_type="true_false", score_category=None, score_metadata=None, score_rationale="rationale", scorer_class_identifier=self.get_identifier(), prompt_request_response_id="mock_id", objective=objective)]
    def validate_return_scores(self, scores: list[Score]):
        assert all(s.score_value in ["true", "false"] for s in scores)

def test_validate_request_raises_on_empty():

    scorer = MockScorer()
    with pytest.raises(ValueError):
        scorer.validate_request(PromptRequestResponse(request_pieces=[]))




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
            objective="task",
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
            objective="task",
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
        objective="task",
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
        objective="task",
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
        objective="task",
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
            objective="task",
        )

        mock_remove_markdown_json.assert_called_once()


@pytest.mark.asyncio
async def test_score_response_async_empty_scorers():
    """Test that score_response_async returns empty list when no scorers provided."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value="test")])

    result = await Scorer.score_response_async(response=response, objective="test task")
    assert result == {"auxiliary_scores": [], "objective_scores": []}


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
    scorer.score_async = AsyncMock(return_value=[])

    result = await Scorer.score_response_async(
        response=response,
        objective_scorer=scorer,
        auxiliary_scorers=[scorer],
        role_filter="assistant",
        objective="test task"
    )
    assert result == {"auxiliary_scores": [], "objective_scores": []}
    scorer.score_async.assert_called()


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
        response=response, auxiliary_scorers=[scorer1, scorer2], role_filter="assistant", objective="test task"
    )

    assert score1_1 in result["auxiliary_scores"]
    assert score2_1 in result["auxiliary_scores"]
    scorer1.score_async.assert_any_call(request_response=response, objective="test task", role_filter="assistant", skip_on_error=True)
    scorer2.score_async.assert_any_call(request_response=response, objective="test task", role_filter="assistant", skip_on_error=True)


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_empty_scorers():
    """Test that score_response_select_first_success_async returns None when no scorers provided."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value="test")])

    result = await Scorer.score_response_multiple_scorers_async(response=response, scorers=[], objective="test task")

    assert result == []


@pytest.mark.asyncio
async def test_score_async_no_matching_role():
    """Test that score_response_select_first_success_async returns None when no pieces match role filter."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="user", original_value="test")])
    scorer = MockScorer()
    result = await scorer.score_async(
        request_response=response, role_filter="assistant", objective="test task"
    )

    assert result == []

@pytest.mark.asyncio
async def test_score_response_async_finds_success():
    """Test that score_response_async returns first successful score."""
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

    result = await Scorer.score_response_multiple_scorers_async(
        response=response, scorers=[scorer1, scorer2], objective="test task"
    )

    # Should return the first successful score (score2)
    assert len(result) == 2
    assert score2 in result

    # scorer1 should be called only once (for piece1)
    assert scorer1.score_async.call_count == 1
    # scorer2 should be called only once (for piece1, returning success)
    assert scorer2.score_async.call_count == 1


@pytest.mark.asyncio
async def test_score_response_success_async_no_success_returns_first():
    """Test that score_response_success_async returns first score when no success found."""
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

    result = await Scorer.score_response_multiple_scorers_async(
        response=response, scorers=[scorer1, scorer2], objective="test task"
    )

    assert score1 in result
    assert score2 in result

    assert scorer1.score_async.call_count == 1
    assert scorer2.score_async.call_count == 1


@pytest.mark.asyncio
async def test_score_response_success_async_parallel_scoring_per_piece():
    """Test that score_response_success_async runs scorers in parallel for each piece."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2")

    response = PromptRequestResponse(request_pieces=[piece1, piece2])

    # Track call order
    call_order = []

    async def mock_score_async_1(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        call_order.append(("scorer1", request_response.request_pieces[0].original_value))
        score = MagicMock(spec=Score)
        score.get_value.return_value = False
        return [score]

    async def mock_score_async_2(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        call_order.append(("scorer2", request_response.request_pieces[0].original_value))
        score = MagicMock(spec=Score)
        score.get_value.return_value = False
        return [score]

    scorer1 = MockScorer()
    scorer1.score_async = mock_score_async_1

    scorer2 = MockScorer()
    scorer2.score_async = mock_score_async_2

    await Scorer.score_response_multiple_scorers_async(
        response=response, scorers=[scorer1, scorer2], objective="test task"
    )

    assert len(call_order) == 2

    assert ("scorer1", "response1") in call_order[:2]
    assert ("scorer2", "response1") in call_order[:2]



@pytest.mark.asyncio
async def test_score_response_async_empty_response():
    """Test score_response_async with empty response."""
    response = PromptRequestResponse(request_pieces=[])

    aux_scorer = MockScorer()
    obj_scorer = MockScorer()

    with pytest.raises(ValueError, match="Empty request pieces"):
        await Scorer.score_response_async(
            response=response, auxiliary_scorers=[aux_scorer], objective_scorer=obj_scorer, objective="test task"
        )


@pytest.mark.asyncio
async def test_score_response_async_no_scorers():
    """Test score_response_async with no scorers provided."""
    response = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value="test")])

    result = await Scorer.score_response_async(
        response=response, auxiliary_scorers=None, objective_scorer=None, objective="test task"
    )

    assert result == {"auxiliary_scores": [], "objective_scores": []}


@pytest.mark.asyncio
async def test_score_response_async_auxiliary_only():
    """Test score_response_async with only auxiliary scorers."""
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

    result = await Scorer.score_response_async(
        response=response, auxiliary_scorers=[aux_scorer1, aux_scorer2], objective_scorer=None, objective="test task"
    )

    # Should have auxiliary scores but no objective scores
    assert len(result["auxiliary_scores"]) == 2
    assert aux_score1 in result["auxiliary_scores"]
    assert aux_score2 in result["auxiliary_scores"]
    assert result["objective_scores"] == []


@pytest.mark.asyncio
async def test_score_response_async_objective_only():
    """Test score_response_async with only objective scorers."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Create mock objective score
    obj_score = MagicMock(spec=Score)
    obj_score.get_value.return_value = True

    # Create mock objective scorer
    obj_scorer = MockScorer()
    obj_scorer.score_async = AsyncMock(return_value=[obj_score])

    result = await Scorer.score_response_async(
        response=response, auxiliary_scorers=None, objective_scorer=obj_scorer, objective="test task"
    )

    # Should have objective score but no auxiliary scores
    assert result["auxiliary_scores"] == []
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score


@pytest.mark.asyncio
async def test_score_response_async_both_types():
    """Test score_response_async with both auxiliary and objective scorers."""
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

    result = await Scorer.score_response_async(
        response=response, auxiliary_scorers=[aux_scorer], objective_scorer=obj_scorer, objective="test task"
    )

    # Should have both types of scores
    assert len(result["auxiliary_scores"]) == 1
    assert result["auxiliary_scores"][0] == aux_score
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score


@pytest.mark.asyncio
async def test_score_response_async_multiple_pieces():
    """Test score_response_async with multiple response pieces."""
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

    result = await Scorer.score_response_async(
        response=response,
        auxiliary_scorers=[aux_scorer1, aux_scorer2],
        objective_scorer=obj_scorer,
        objective="test task",
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
async def test_score_response_async_role_filter():
    """Test score_response_async with different role filters."""
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

    async def track_aux_score(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        aux_scored_pieces.append(request_response.request_pieces[0])
        return [aux_score]

    async def track_obj_score(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        obj_scored_pieces.append(request_response.request_pieces[0])
        return [obj_score]

    aux_scorer = MockScorer()
    aux_scorer.score_async = track_aux_score

    obj_scorer = MockScorer()
    obj_scorer.score_async = track_obj_score

    result = await Scorer.score_response_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorer=obj_scorer,
        role_filter="assistant",
        objective="test task",
    )

    # Should only score assistant pieces
    assert len(aux_scored_pieces) == 1
    assert aux_scored_pieces[0].role == "assistant"
    assert len(obj_scored_pieces) == 1
    assert obj_scored_pieces[0].role == "assistant"

    assert len(result["auxiliary_scores"]) == 1
    assert len(result["objective_scores"]) == 1


@pytest.mark.asyncio
async def test_score_response_async_skip_on_error_true():
    """Test score_response_async skips error pieces when skip_on_error=True."""
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

    result = await Scorer.score_response_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorer=obj_scorer,
        objective="test task",
        skip_on_error=True,
    )

    # Should only score the non-error piece
    assert len(result["auxiliary_scores"]) == 1
    assert len(result["objective_scores"]) == 1

    # Verify only non-error piece was scored
    aux_scorer.score_async.assert_called_once()
    obj_scorer.score_async.assert_called_once()


@pytest.mark.asyncio
async def test_score_response_async_skip_on_error_false():
    """Test score_response_async includes error pieces when skip_on_error=False."""
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

    result = await Scorer.score_response_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorer=obj_scorer,
        objective="test task",
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
async def test_score_response_async_objective_failure():
    """Test score_response_async when no objective succeeds."""
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

    result = await Scorer.score_response_async(
        response=response, auxiliary_scorers=None, objective_scorer=obj_scorer1, objective="test task"
    )

    # Should return the first score as failure indicator
    assert result["auxiliary_scores"] == []
    assert len(result["objective_scores"]) == 1
    assert result["objective_scores"][0] == obj_score1


@pytest.mark.asyncio
async def test_score_response_async_concurrent_execution():
    """Test that auxiliary and objective scoring happen concurrently."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    # Track call order to verify concurrent execution
    call_order = []

    async def mock_aux_score_async(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        call_order.append("aux_start")
        # Simulate some async work
        await asyncio.sleep(0.01)
        call_order.append("aux_end")
        return [MagicMock(spec=Score)]

    async def mock_obj_score_async(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
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

    await Scorer.score_response_async(
        response=response, auxiliary_scorers=[aux_scorer], objective_scorer=obj_scorer, objective="test task"
    )

    # Both should start before either finishes (concurrent execution)
    assert call_order.index("aux_start") < call_order.index("obj_end")
    assert call_order.index("obj_start") < call_order.index("aux_end")


@pytest.mark.asyncio
async def test_score_response_async_empty_lists():
    """Test score_response_async with empty scorer lists."""
    piece = PromptRequestPiece(role="assistant", original_value="response")
    response = PromptRequestResponse(request_pieces=[piece])

    result = await Scorer.score_response_async(
        response=response, auxiliary_scorers=[], objective_scorer=None, objective="test task"
    )

    assert result == {"auxiliary_scores": [], "objective_scores": []}



@pytest.mark.asyncio
async def test_score_response_async_mixed_roles():
    """Test score_response_async filters roles correctly."""
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

    async def track_aux_score(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        aux_scored_pieces.append(request_response.request_pieces[0])
        return [aux_score]

    async def track_obj_score(request_response: PromptRequestResponse, **kwargs) -> list[Score]:
        obj_scored_pieces.append(request_response.request_pieces[0])
        return [obj_score]

    aux_scorer = MockScorer()
    aux_scorer.score_async = track_aux_score

    obj_scorer = MockScorer()
    obj_scorer.score_async = track_obj_score

    result = await Scorer.score_response_async(
        response=response,
        auxiliary_scorers=[aux_scorer],
        objective_scorer=obj_scorer,
        role_filter="assistant",
        objective="test task",
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
