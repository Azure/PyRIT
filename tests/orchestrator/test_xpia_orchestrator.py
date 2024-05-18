# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.orchestrator import (
    XPIATestOrchestrator,
    XPIAOrchestrator,
    XPIAManualProcessingOrchestrator,
)
import pytest

from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.score import Score, Scorer
from tests.mocks import get_memory_interface, MockPromptTarget


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def attack_setup_target(memory_interface) -> PromptTarget:
    return MockPromptTarget(memory=memory_interface)


@pytest.fixture
def processing_target() -> PromptTarget:
    return MockPromptTarget()


@pytest.fixture
def success_scorer() -> Scorer:
    mock_score = MagicMock(Score)
    mock_score.score_value = True
    mock_score.score_type = "true_false"
    mock_score.get_value.return_value = True

    mock_scorer = MagicMock(Scorer)
    mock_scorer.scorer_type = "true_false"
    mock_scorer.score_text_async = AsyncMock(return_value=[mock_score])
    return mock_scorer


@pytest.mark.asyncio
async def test_xpia_orchestrator_execute_no_scorer(attack_setup_target):
    def processing_callback():
        return_request_response_obj = Mock()
        return_response_piece = Mock()
        return_response_piece.converted_prompt_text = "test"
        return_request_response_obj.request_pieces = [return_response_piece]
        return return_request_response_obj

    xpia_orchestrator = XPIAOrchestrator(
        attack_content="test",
        attack_setup_target=attack_setup_target,
        processing_callback=processing_callback,
    )

    xpia_operation = await xpia_orchestrator.execute_async()
    assert xpia_operation is None


@pytest.mark.asyncio
async def test_xpia_orchestrator_execute(attack_setup_target, success_scorer):
    def processing_callback():
        return_request_response_obj = Mock()
        return_response_piece = Mock()
        return_response_piece.converted_prompt_text = "test"
        return_request_response_obj.request_pieces = [return_response_piece]
        return return_request_response_obj

    xpia_orchestrator = XPIAOrchestrator(
        attack_content="test",
        attack_setup_target=attack_setup_target,
        scorer=success_scorer,
        processing_callback=processing_callback,
    )
    score = await xpia_orchestrator.execute_async()
    assert score.score_value
    assert success_scorer.score_text_async.called_once


@pytest.mark.asyncio
async def test_xpia_manual_processing_orchestrator_execute(attack_setup_target, success_scorer, monkeypatch):
    # Mocking user input to be "test"
    monkeypatch.setattr("builtins.input", lambda _: "test")
    xpia_orchestrator = XPIAManualProcessingOrchestrator(
        attack_content="test",
        attack_setup_target=attack_setup_target,
        scorer=success_scorer,
    )
    score = await xpia_orchestrator.execute_async()
    assert score.score_value
    assert success_scorer.score_text_async.called_once


@pytest.mark.asyncio
async def test_xpia_test_orchestrator_execute(attack_setup_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt_async") as mock_send_to_processing_target:
        xpia_orchestrator = XPIATestOrchestrator(
            attack_content="test",
            processing_prompt="some instructions and the required <test>",
            processing_target=processing_target,
            attack_setup_target=attack_setup_target,
            scorer=success_scorer,
        )
        score = await xpia_orchestrator.execute_async()
        assert score.score_value
        assert success_scorer.score_text_async.called_once
        assert mock_send_to_processing_target.called_once


@pytest.mark.asyncio
async def test_xpia_orchestrator_process_async(attack_setup_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt_async") as mock_send_to_processing_target:
        with patch.object(processing_target, "send_prompt_async") as mock_send_async_to_processing_target:
            mock_send_to_processing_target.side_effect = NotImplementedError()
            xpia_orchestrator = XPIATestOrchestrator(
                attack_content="test",
                processing_prompt="some instructions and the required <test>",
                processing_target=processing_target,
                attack_setup_target=attack_setup_target,
                scorer=success_scorer,
            )
            score = await xpia_orchestrator.execute_async()
            assert score.score_value
            assert success_scorer.score_text_async.called_once
            assert mock_send_to_processing_target.called_once
            assert mock_send_async_to_processing_target.assert_called_once
