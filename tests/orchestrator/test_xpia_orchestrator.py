# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from pyrit.interfaces import SupportTextClassification
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.orchestrator.xpia_orchestrator import XPIATestOrchestrator, XPIAOrchestrator, XPIAManualProcessingOrchestrator
import pytest

from unittest.mock import Mock, patch

from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.models import Score
from tests.mocks import get_memory_interface, MockPromptTarget


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def prompt_target(memory_interface) -> PromptTarget:
    return MockPromptTarget(memory=memory_interface)


@pytest.fixture
def processing_target() -> PromptTarget:
    return MockPromptTarget()


@pytest.fixture
def success_scorer() -> SupportTextClassification:
    scorer = Mock()
    scorer.score_text = Mock(return_value=Score(score_type="bool", score_value=True))
    return scorer


def test_xpia_orchestrator_execute_no_scorer(prompt_target):
    def processing_callback():
        return_request_response_obj = Mock()
        return_response_piece = Mock()
        return_response_piece.converted_prompt_text = "test"
        return_request_response_obj.request_pieces = [return_response_piece]
        return return_request_response_obj

    xpia_orchestrator = XPIAOrchestrator(
        attack_content="test",
        prompt_target=prompt_target,
        processing_callback=processing_callback,
    )
    assert xpia_orchestrator.execute() is None


def test_xpia_orchestrator_execute(prompt_target, success_scorer):
    def processing_callback():
        return_request_response_obj = Mock()
        return_response_piece = Mock()
        return_response_piece.converted_prompt_text = "test"
        return_request_response_obj.request_pieces = [return_response_piece]
        return return_request_response_obj

    xpia_orchestrator = XPIAOrchestrator(
        attack_content="test",
        prompt_target=prompt_target,
        scorer=success_scorer,
        processing_callback=processing_callback,
    )
    score = xpia_orchestrator.execute()
    assert score.score_value
    assert success_scorer.score_text.called_once


def test_xpia_manual_processing_orchestrator_execute(prompt_target, success_scorer, monkeypatch):
    # Mocking user input to be "test"
    monkeypatch.setattr('builtins.input', lambda _: "test")
    xpia_orchestrator = XPIAManualProcessingOrchestrator(
        attack_content="test",
        prompt_target=prompt_target,
        scorer=success_scorer,
    )
    score = xpia_orchestrator.execute()
    assert score.score_value
    assert success_scorer.score_text.called_once


def test_xpia_test_orchestrator_execute(prompt_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt") as mock_send_to_processing_target:
        xpia_orchestrator = XPIATestOrchestrator(
            attack_content="test",
            processing_prompt="some instructions and the required <test>",
            processing_target=processing_target,
            prompt_target=prompt_target,
            scorer=success_scorer,
        )
        score = xpia_orchestrator.execute()
        assert score.score_value
        assert success_scorer.score_text.called_once
        assert mock_send_to_processing_target.called_once


def test_xpia_orchestrator_process_async(prompt_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt") as mock_send_to_processing_target:
        with patch.object(processing_target, "send_prompt_async") as mock_send_async_to_processing_target:
            mock_send_to_processing_target.side_effect = NotImplementedError()
            xpia_orchestrator = XPIATestOrchestrator(
                attack_content="test",
                processing_prompt="some instructions and the required <test>",
                processing_target=processing_target,
                prompt_target=prompt_target,
                scorer=success_scorer,
            )
            score = xpia_orchestrator.execute()
            assert score.score_value
            assert success_scorer.score_text.called_once
            assert mock_send_to_processing_target.called_once
            assert mock_send_async_to_processing_target.assert_called_once
