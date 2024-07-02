# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from unittest.mock import AsyncMock, MagicMock
import pytest

from pyrit.memory import DuckDBMemory
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.score import Score

from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest, NormalizerRequestPiece
from tests.mocks import MockPromptTarget


@pytest.fixture
def mock_target() -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    file_memory = DuckDBMemory(db_path=":memory:")
    return MockPromptTarget(memory=file_memory)


@pytest.mark.asyncio
async def test_send_prompt_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    await orchestrator.send_prompts_async(prompt_list=["Hello"])
    assert mock_target.prompt_sent == ["Hello"]


@pytest.mark.asyncio
async def test_send_prompts_async_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    await orchestrator.send_prompts_async(prompt_list=["Hello"])
    assert mock_target.prompt_sent == ["Hello"]


@pytest.mark.asyncio
async def test_send_multiple_prompts_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    await orchestrator.send_prompts_async(prompt_list=["Hello", "my", "name"])
    assert mock_target.prompt_sent == ["Hello", "my", "name"]


@pytest.mark.asyncio
async def test_send_prompts_b64_converter(mock_target: MockPromptTarget):
    converter = Base64Converter()
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, prompt_converters=[converter])

    await orchestrator.send_prompts_async(prompt_list=["Hello"])
    assert mock_target.prompt_sent == ["SGVsbG8="]


@pytest.mark.asyncio
async def test_send_prompts_multiple_converters(mock_target: MockPromptTarget):
    b64_converter = Base64Converter()
    join_converter = StringJoinConverter(join_value="_")

    # This should base64 encode the prompt and then join the characters with an underscore
    converters = [b64_converter, join_converter]

    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, prompt_converters=converters)

    await orchestrator.send_prompts_async(prompt_list=["Hello"])
    assert mock_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_normalizer_requests_async(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)
    orchestrator._prompt_normalizer = AsyncMock()
    orchestrator._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=None)

    with tempfile.NamedTemporaryFile(suffix=".png") as f:

        f.write(b"test")
        f.flush()
        req = NormalizerRequestPiece(
            request_converters=[Base64Converter()],
            prompt_data_type="image_path",
            prompt_value=f.name,
        )

        await orchestrator.send_normalizer_requests_async(prompt_request_list=[NormalizerRequest(request_pieces=[req])])
        assert orchestrator._prompt_normalizer.send_prompt_batch_to_target_async.called


@pytest.mark.asyncio
async def test_send_normalizer_request_scores_async(mock_target: MockPromptTarget):
    # Set up mocks and return values
    scorer = AsyncMock()
    scorer._memory = MagicMock()
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, scorers=[scorer])
    orchestrator._prompt_normalizer = AsyncMock()

    request = PromptRequestPiece(
        role="user", original_value="test request", orchestrator_identifier=orchestrator.get_identifier()
    )
    response = PromptRequestPiece(
        role="assistant", original_value="test response to score", orchestrator_identifier=orchestrator.get_identifier()
    )

    request_pieces = [request, response]
    orchestrator._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
        return_value=[piece.to_prompt_request_response() for piece in request_pieces]
    )

    orchestrator._memory = MagicMock()
    orchestrator._memory.get_prompt_request_pieces_by_id = MagicMock(return_value=request_pieces)  # type: ignore

    # Call directly into function to send prompts
    req = NormalizerRequestPiece(
        request_converters=[],
        prompt_data_type="text",
        prompt_value="test request",
    )

    await orchestrator.send_normalizer_requests_async(prompt_request_list=[NormalizerRequest(request_pieces=[req])])
    assert orchestrator._prompt_normalizer.send_prompt_batch_to_target_async.called
    scorer.score_async.assert_called_once_with(request_response=response)

    # Check that sending another prompt request scores the appropriate pieces
    scorer.score_async.reset_mock()
    response2 = PromptRequestPiece(
        role="assistant",
        original_value="test response to score 2",
        orchestrator_identifier=orchestrator.get_identifier(),
    )

    request_pieces = [request, response2]
    orchestrator._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
        return_value=[piece.to_prompt_request_response() for piece in request_pieces]
    )
    orchestrator._memory.get_prompt_request_pieces_by_id = MagicMock(return_value=request_pieces)  # type: ignore

    await orchestrator.send_normalizer_requests_async(prompt_request_list=[NormalizerRequest(request_pieces=[req])])
    scorer.score_async.assert_called_once_with(request_response=response2)


def test_sendprompts_orchestrator_sets_target_memory(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)
    assert orchestrator._memory is mock_target._memory


def test_send_prompt_to_identifier(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    d = orchestrator.get_identifier()
    assert d["id"]
    assert d["__type__"] == "PromptSendingOrchestrator"
    assert d["__module__"] == "pyrit.orchestrator.prompt_sending_orchestrator"


def test_orchestrator_get_memory(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    request = PromptRequestPiece(
        role="user",
        original_value="test",
        orchestrator_identifier=orchestrator.get_identifier(),
    ).to_prompt_request_response()

    orchestrator._memory.add_request_response_to_memory(request=request)

    entries = orchestrator.get_memory()
    assert entries
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_orchestrator_get_score_memory(mock_target: MockPromptTarget):
    scorer = AsyncMock()
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, scorers=[scorer])

    request = PromptRequestPiece(
        role="user",
        original_value="test",
        orchestrator_identifier=orchestrator.get_identifier(),
    )

    score = Score(
        score_type="float_scale",
        score_value=str(1),
        score_value_description=None,
        score_category="mock",
        score_metadata=None,
        score_rationale=None,
        scorer_class_identifier=orchestrator.get_identifier(),
        prompt_request_response_id=request.id,
    )

    orchestrator._memory.add_request_pieces_to_memory(request_pieces=[request])
    orchestrator._memory.add_scores_to_memory(scores=[score])

    scores = orchestrator.get_score_memory()
    assert len(scores) == 1
    assert scores[0].prompt_request_response_id == request.id
