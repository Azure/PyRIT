# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from pyrit.interfaces import SupportTextClassification
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.orchestrator.xpia_orchestrator import XPIATestOrchestrator
import pytest

from unittest.mock import Mock, patch

from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.models import Score
from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


class MockPromptTarget(PromptTarget):
    def __init__(
        self,
        *,
        memory: MemoryInterface = None,
    ) -> None:
        super().__init__(memory=memory)

    def send_prompt(
        self,
        *,
        normalized_prompt: str,
        conversation_id: str,
        normalizer_id: str,
    ) -> str:
        pass

    async def send_prompt_async(
        self,
        *,
        normalized_prompt: str,
        conversation_id: str,
        normalizer_id: str,
    ) -> str:
        pass


@pytest.fixture
def prompt_target(memory_interface) -> PromptTarget:
    return MockPromptTarget(memory=memory_interface)


@pytest.fixture
def processing_target() -> PromptTarget:
    return MockPromptTarget()


@pytest.fixture
def success_scorer() -> SupportTextClassification:
    scorer = Mock()
    scorer.score_text = Mock(
        return_value=Score(
            score_type="bool",
            score_value=True
        )
    )
    return scorer


def test_xpia_orchestrator_process(prompt_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt") as mock_send_to_processing_target:
        xpia_orchestrator = XPIATestOrchestrator(
            attack_content="test",
            processing_prompt="some instructions and the required <test>",
            processing_target=processing_target,
            prompt_target=prompt_target,
            scorer=success_scorer,
        )
        score = xpia_orchestrator.process()
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
            score = xpia_orchestrator.process()
            assert score.score_value
            assert success_scorer.score_text.called_once
            assert mock_send_to_processing_target.called_once
            assert mock_send_async_to_processing_target.assert_called_once

