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
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer, ScorerPromptValidator, TrueFalseScorer


@pytest.fixture
def good_json():
    return '{"score_value": "1", "rationale": "Valid response"}'


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


class MockScorer(TrueFalseScorer):
    def __init__(self):
        super().__init__(validator=DummyValidator())

    async def _score_async(
        self, request_response: PromptRequestResponse, *, objective: Optional[str] = None
    ) -> list[Score]:
        return [
            Score(
                score_value="true",
                score_value_description="desc",
                score_type="true_false",
                score_category=None,
                score_metadata=None,
                score_rationale="rationale",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id="mock_id",
                objective=objective,
            )
        ]

    async def _score_piece_async(
        self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        return [
            Score(
                score_value="true",
                score_value_description="desc",
                score_type="true_false",
                score_category=None,
                score_metadata=None,
                score_rationale="rationale",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id="mock_id",
                objective=objective,
            )
        ]

    def validate_return_scores(self, scores: list[Score]):
        assert all(s.score_value in ["true", "false"] for s in scores)


class SelectiveValidator(ScorerPromptValidator):
    """Validator that only supports text pieces, not images."""

    def __init__(self, *, enforce_all_pieces_valid: bool = False):
        super().__init__(
            supported_data_types=["text"],
            enforce_all_pieces_valid=enforce_all_pieces_valid,
        )


class MockFloatScorer(Scorer):
    """Mock scorer that tracks which pieces were scored."""

    def __init__(self, *, validator: ScorerPromptValidator):
        super().__init__(validator=validator)
        self.scored_piece_ids: list[str] = []

    async def _score_piece_async(
        self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        # Track which pieces get scored
        self.scored_piece_ids.append(str(request_piece.id))

        return [
            Score(
                score_value="0.5",
                score_value_description="Test score",
                score_type="float_scale",
                score_category=None,
                score_metadata=None,
                score_rationale="Test rationale",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_piece.id or "test-id",
                objective=objective,
            )
        ]

    def validate_return_scores(self, scores: list[Score]):
        for score in scores:
            assert 0 <= float(score.score_value) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_json", [BAD_JSON, KEY_ERROR_JSON, KEY_ERROR2_JSON])
async def test_scorer_send_chat_target_async_bad_json_exception_retries(bad_json: str):
    chat_target = MagicMock(PromptChatTarget)
    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=bad_json, conversation_id="test-convo")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)
    scorer = MockScorer()
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
async def test_scorer_score_value_with_llm_use_provided_attack_identifier(good_json):
    scorer = MockScorer()

    prompt_response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json, conversation_id="test-convo")]
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
        objective="task",
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

    prompt_response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json, conversation_id="test-convo")]
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
    assert not set_sys_prompt_args["attack_identifier"]


@pytest.mark.asyncio
async def test_scorer_send_chat_target_async_good_response(good_json):

    chat_target = MagicMock(PromptChatTarget)

    good_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json, conversation_id="test-convo")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=good_json_resp)

    scorer = MockScorer()

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
        request_pieces=[PromptRequestPiece(role="assistant", original_value=good_json, conversation_id="test-convo")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=good_json_resp)

    scorer = MockScorer()

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


def test_scorer_extract_task_from_response(patch_central_database):
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

        extracted_task = scorer._extract_objective_from_response(response_piece.to_prompt_request_response())
        assert "User's question about the universe" in extracted_task


@pytest.mark.asyncio
async def test_scorer_score_responses_batch_async(patch_central_database):
    """
    Test that score_responses_batch_async filters to only assistant pieces,
    calls score_prompts_with_tasks_batch_async, and returns results.
    """
    scorer = MockScorer()

    with patch.object(scorer, "score_async", new_callable=AsyncMock) as mock_score_async:
        fake_scores = [MagicMock(), MagicMock()]
        mock_score_async.return_value = fake_scores

        user_req = PromptRequestPiece(role="user", original_value="Hello user", sequence=1).to_prompt_request_response()
        assistant_resp = PromptRequestPiece(
            role="assistant", original_value="Hello from assistant", sequence=2
        ).to_prompt_request_response()

        results = await scorer.score_prompts_batch_async(
            request_responses=[user_req, assistant_resp], batch_size=10, infer_objective_from_request=True
        )

        # Verify mock_score_async was called twice
        assert mock_score_async.call_count == 2

        # Get the call_args for the first call
        _, first_call_kwargs = mock_score_async.call_args_list[0]

        assert "request_response" in first_call_kwargs
        assert "objective" in first_call_kwargs
        assert "infer_objective_from_request" in first_call_kwargs
        assert first_call_kwargs["request_response"] == user_req

        assert fake_scores[0] in results
        assert len(fake_scores) == 2


@pytest.mark.asyncio
async def test_score_response_async_empty_scorers():
    """Test that score_response_async returns empty list when no scorers provided."""
    response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="test", conversation_id="test-convo")]
    )

    result = await Scorer.score_response_async(response=response, objective="test task")
    assert result == {"auxiliary_scores": [], "objective_scores": []}


@pytest.mark.asyncio
async def test_score_response_async_no_matching_role():
    """Test that score_response_async returns empty list when no pieces match role filter."""
    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value="test1", conversation_id="test-convo"),
            PromptRequestPiece(role="user", original_value="test2", conversation_id="test-convo"),
        ]
    )

    scorer = MockScorer()
    scorer.score_async = AsyncMock(return_value=[])

    result = await Scorer.score_response_async(
        response=response,
        objective_scorer=scorer,
        auxiliary_scorers=[scorer],
        role_filter="assistant",
        objective="test task",
    )
    assert result == {"auxiliary_scores": [], "objective_scores": []}
    scorer.score_async.assert_called()


@pytest.mark.asyncio
async def test_score_response_async_parallel_execution():
    """Test that score_response_async runs all scorers in parallel on all filtered pieces."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1", conversation_id="test-convo")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2", conversation_id="test-convo")
    piece3 = PromptRequestPiece(role="assistant", original_value="user input", conversation_id="test-convo")

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
    scorer1.score_async.assert_any_call(
        request_response=response, objective="test task", role_filter="assistant", skip_on_error_result=True
    )
    scorer2.score_async.assert_any_call(
        request_response=response, objective="test task", role_filter="assistant", skip_on_error_result=True
    )


@pytest.mark.asyncio
async def test_score_response_select_first_success_async_empty_scorers():
    """Test that score_response_select_first_success_async returns None when no scorers provided."""
    response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="test", conversation_id="test-convo")]
    )

    result = await Scorer.score_response_multiple_scorers_async(response=response, scorers=[], objective="test task")

    assert result == []


@pytest.mark.asyncio
async def test_score_async_no_matching_role():
    """Test that score_response_select_first_success_async returns None when no pieces match role filter."""
    response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", original_value="test", conversation_id="test-convo")]
    )
    scorer = MockScorer()
    result = await scorer.score_async(request_response=response, role_filter="assistant", objective="test task")

    assert result == []


@pytest.mark.asyncio
async def test_score_response_async_finds_success():
    """Test that score_response_async returns first successful score."""
    piece1 = PromptRequestPiece(role="assistant", original_value="response1", conversation_id="test-convo")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2", conversation_id="test-convo")

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
    piece1 = PromptRequestPiece(role="assistant", original_value="response1", conversation_id="test-convo")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2", conversation_id="test-convo")

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
    piece1 = PromptRequestPiece(role="assistant", original_value="response1", conversation_id="test-convo")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2", conversation_id="test-convo")

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
    piece1 = PromptRequestPiece(role="assistant", original_value="response1", conversation_id="test-convo")
    piece2 = PromptRequestPiece(role="assistant", original_value="response2", conversation_id="test-convo")
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
async def test_score_response_async_skip_on_error_true():
    """Test score_response_async skips error pieces when skip_on_error_result=True."""
    piece1 = PromptRequestPiece(role="assistant", original_value="good response", conversation_id="test-convo")
    piece2 = PromptRequestPiece(
        role="assistant", original_value="error", response_error="blocked", conversation_id="test-convo"
    )
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
        skip_on_error_result=True,
    )

    # Should only score the non-error piece
    assert len(result["auxiliary_scores"]) == 1
    assert len(result["objective_scores"]) == 1

    # Verify only non-error piece was scored
    aux_scorer.score_async.assert_called_once()
    obj_scorer.score_async.assert_called_once()


@pytest.mark.asyncio
async def test_score_response_async_skip_on_error_false():
    """Test score_response_async includes error pieces when skip_on_error_result=False."""
    piece1 = PromptRequestPiece(role="assistant", original_value="good response", conversation_id="test-convo")
    piece2 = PromptRequestPiece(
        role="assistant", original_value="error", response_error="blocked", conversation_id="test-convo"
    )
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
        skip_on_error_result=False,
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


@pytest.mark.asyncio
async def test_get_supported_pieces_filters_unsupported_data_types(patch_central_database):
    """Test that _get_supported_pieces only returns pieces with supported data types."""
    validator = SelectiveValidator(enforce_all_pieces_valid=False)
    scorer = MockFloatScorer(validator=validator)

    # Verify validator is configured correctly
    assert "text" in validator._supported_data_types
    assert (
        "image_path" not in validator._supported_data_types
        or len([dt for dt in validator._supported_data_types if dt != "text"]) == 0
    )

    # Create a response with mixed data types
    text_piece = PromptRequestPiece(
        role="assistant",
        original_value="text response",
        converted_value_data_type="text",
        id="text-1",
        conversation_id="test-convo",
    )
    image_piece = PromptRequestPiece(
        role="assistant",
        original_value="image.png",
        converted_value_data_type="image_path",
        id="image-1",
        conversation_id="test-convo",
    )
    audio_piece = PromptRequestPiece(
        role="assistant",
        original_value="audio.wav",
        converted_value_data_type="audio_path",
        id="audio-1",
        conversation_id="test-convo",
    )

    # Verify validator filtering works
    assert validator.is_request_piece_supported(text_piece) is True
    assert validator.is_request_piece_supported(image_piece) is False
    assert validator.is_request_piece_supported(audio_piece) is False

    response = PromptRequestResponse(request_pieces=[text_piece, image_piece, audio_piece])

    # Score the response
    scores = await scorer.score_async(response)

    # Should only score the text piece
    assert len(scorer.scored_piece_ids) == 1
    assert scorer.scored_piece_ids[0] == "text-1"
    assert len(scores) == 1
    assert scores[0].prompt_request_response_id == "text-1"


@pytest.mark.asyncio
async def test_unsupported_pieces_ignored_when_enforce_all_pieces_valid_false(patch_central_database):
    """Test that unsupported pieces don't cause errors when enforce_all_pieces_valid=False."""
    validator = SelectiveValidator(enforce_all_pieces_valid=False)
    scorer = MockFloatScorer(validator=validator)

    # Create a response with only unsupported types and one supported
    text_piece = PromptRequestPiece(
        role="assistant",
        original_value="text response",
        converted_value_data_type="text",
        id="text-1",
        conversation_id="test-convo",
    )
    image_piece = PromptRequestPiece(
        role="assistant",
        original_value="image.png",
        converted_value_data_type="image_path",
        id="image-1",
        conversation_id="test-convo",
    )

    response = PromptRequestResponse(request_pieces=[image_piece, text_piece])

    # Should not raise an error, just skip the image piece
    scores = await scorer.score_async(response)

    assert len(scores) == 1
    assert len(scorer.scored_piece_ids) == 1
    assert scorer.scored_piece_ids[0] == "text-1"


@pytest.mark.asyncio
async def test_all_unsupported_pieces_raises_error(patch_central_database):
    """Test that having no supported pieces raises a clear error."""
    validator = SelectiveValidator(enforce_all_pieces_valid=False)
    scorer = MockFloatScorer(validator=validator)

    # Create a response with only unsupported types
    image_piece = PromptRequestPiece(
        role="assistant",
        original_value="image.png",
        converted_value_data_type="image_path",
        id="image-1",
        conversation_id="test-convo",
    )
    audio_piece = PromptRequestPiece(
        role="assistant",
        original_value="audio.wav",
        converted_value_data_type="audio_path",
        id="audio-1",
        conversation_id="test-convo",
    )

    response = PromptRequestResponse(request_pieces=[image_piece, audio_piece])

    # Should raise error from validator because no valid pieces to score
    with pytest.raises(ValueError, match="There are no valid pieces to score"):
        await scorer.score_async(response)

    # No pieces should have been scored
    assert len(scorer.scored_piece_ids) == 0


@pytest.mark.asyncio
async def test_true_false_scorer_uses_supported_pieces_only(patch_central_database):
    """Test that TrueFalseScorer also uses _get_supported_pieces via base implementation."""
    validator = SelectiveValidator(enforce_all_pieces_valid=False)

    class TestTrueFalseScorer(TrueFalseScorer):
        def __init__(self):
            super().__init__(validator=validator)
            self.scored_piece_ids = []

        async def _score_piece_async(
            self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
        ) -> list[Score]:
            self.scored_piece_ids.append(request_piece.id)
            return [
                Score(
                    score_value="true",
                    score_value_description="Test",
                    score_type="true_false",
                    score_category=None,
                    score_metadata=None,
                    score_rationale="Test",
                    scorer_class_identifier=self.get_identifier(),
                    prompt_request_response_id=request_piece.id or "test-id",
                    objective=objective,
                )
            ]

    scorer = TestTrueFalseScorer()

    # Create mixed response
    text_piece = PromptRequestPiece(
        role="assistant",
        original_value="text",
        converted_value_data_type="text",
        id="text-1",
        conversation_id="test-convo",
    )
    image_piece = PromptRequestPiece(
        role="assistant",
        original_value="image.png",
        converted_value_data_type="image_path",
        id="image-1",
        conversation_id="test-convo",
    )

    response = PromptRequestResponse(request_pieces=[text_piece, image_piece])

    # Score the response
    scores = await scorer.score_async(response)

    # Should only score the text piece
    assert len(scorer.scored_piece_ids) == 1
    assert scorer.scored_piece_ids[0] == "text-1"
    # TrueFalseScorer aggregates to single score
    assert len(scores) == 1
    assert scores[0].score_value == "true"


@pytest.mark.asyncio
async def test_base_scorer_score_async_implementation(patch_central_database):
    """Test that the base Scorer._score_async implementation works correctly."""
    validator = SelectiveValidator(enforce_all_pieces_valid=False)
    scorer = MockFloatScorer(validator=validator)

    # Create response with multiple supported pieces
    text_piece1 = PromptRequestPiece(
        role="assistant",
        original_value="text 1",
        converted_value_data_type="text",
        id="text-1",
        conversation_id="test-convo",
    )
    text_piece2 = PromptRequestPiece(
        role="assistant",
        original_value="text 2",
        converted_value_data_type="text",
        id="text-2",
        conversation_id="test-convo",
    )

    response = PromptRequestResponse(request_pieces=[text_piece1, text_piece2])

    # Score the response
    scores = await scorer.score_async(response)

    # Should score both pieces
    assert len(scorer.scored_piece_ids) == 2
    assert "text-1" in scorer.scored_piece_ids
    assert "text-2" in scorer.scored_piece_ids
    assert len(scores) == 2
