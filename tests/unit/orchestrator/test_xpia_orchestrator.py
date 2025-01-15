# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models import Score
from pyrit.orchestrator import (
    XPIAManualProcessingOrchestrator,
    XPIAOrchestrator,
    XPIATestOrchestrator,
)
from pyrit.score import Scorer


@pytest.fixture
def attack_setup_target(patch_central_database) -> MockPromptTarget:
    return MockPromptTarget()


@pytest.fixture
def processing_target(patch_central_database) -> MockPromptTarget:
    return MockPromptTarget()


@pytest.fixture
def success_scorer() -> Scorer:
    mock_score = MagicMock(Score)
    mock_score.score_value = "True"
    mock_score.score_type = "true_false"
    mock_score.get_value.return_value = True

    mock_scorer = MagicMock(Scorer)
    mock_scorer.scorer_type = "true_false"
    mock_scorer.score_text_async = AsyncMock(return_value=[mock_score])
    return mock_scorer


@pytest.mark.asyncio
async def test_xpia_orchestrator_execute_no_scorer(attack_setup_target):
    async def processing_callback():
        return_response_piece = AsyncMock()
        return_response_piece.converted_prompt_text = "test_converted_text"
        return_request_response_obj = AsyncMock()
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
    async def processing_callback():
        return_response_piece = AsyncMock()
        return_response_piece.converted_prompt_text = "test_converted_text"
        return_request_response_obj = AsyncMock()
        return_request_response_obj.request_pieces = [return_response_piece]

        return return_request_response_obj

    xpia_orchestrator = XPIAOrchestrator(
        attack_content="test",
        attack_setup_target=attack_setup_target,
        scorer=success_scorer,
        processing_callback=processing_callback,
    )
    score = await xpia_orchestrator.execute_async()
    assert score.get_value()
    success_scorer.score_text_async.assert_called_once()


@pytest.mark.asyncio
@patch(
    "pyrit.common.default_values.get_non_required_value",
    return_value='{"op_name": "dummy_op"}',
)
async def test_xpia_orchestrator_execute_with_memory_labels(
    mock_get_non_required_value, attack_setup_target, success_scorer
):
    async def processing_callback():
        return_response_piece = AsyncMock()
        return_response_piece.converted_prompt_text = "test_converted_text"
        return_request_response_obj = AsyncMock()
        return_request_response_obj.request_pieces = [return_response_piece]

        return return_request_response_obj

    xpia_orchestrator = XPIAOrchestrator(
        attack_content="test",
        attack_setup_target=attack_setup_target,
        scorer=success_scorer,
        processing_callback=processing_callback,
    )

    score = await xpia_orchestrator.execute_async()

    entries = xpia_orchestrator.get_memory()
    assert len(entries) == 2
    assert entries[0].labels == {"op_name": "dummy_op"}
    assert score.get_value()
    success_scorer.score_text_async.assert_called_once()


@pytest.mark.asyncio
@patch.object(XPIAManualProcessingOrchestrator, "_input_async", new_callable=AsyncMock, return_value="test")
async def test_xpia_manual_processing_orchestrator_execute(mock_input_async, attack_setup_target, success_scorer):
    xpia_orchestrator = XPIAManualProcessingOrchestrator(
        attack_content="test",
        attack_setup_target=attack_setup_target,
        scorer=success_scorer,
    )

    score = await xpia_orchestrator.execute_async()

    assert score.get_value()
    success_scorer.score_text_async.assert_called_once()
    mock_input_async.assert_awaited_once()


@pytest.mark.asyncio
@patch(
    "pyrit.common.default_values.get_non_required_value",
    return_value='{"op_name": "dummy_op"}',
)
@patch.object(XPIAManualProcessingOrchestrator, "_input_async", new_callable=AsyncMock, return_value="test")
async def test_xpia_manual_processing_orchestrator_execute_with_memory_labels(
    mock_input_async, mock_get_non_required_value, attack_setup_target, success_scorer
):
    xpia_orchestrator = XPIAManualProcessingOrchestrator(
        attack_content="test", attack_setup_target=attack_setup_target, scorer=success_scorer
    )

    score = await xpia_orchestrator.execute_async()
    entries = xpia_orchestrator.get_memory()
    assert len(entries) == 2
    assert entries[0].labels == {"op_name": "dummy_op"}
    assert score.get_value()
    success_scorer.score_text_async.assert_called_once()
    mock_input_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_xpia_test_orchestrator_execute(attack_setup_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt_async", new_callable=AsyncMock) as mock_send_to_processing_target:
        mock_send_to_processing_target.return_value = AsyncMock(
            request_pieces=[AsyncMock(converted_value="mocked_processing_response")]
        )

        with patch.object(
            XPIATestOrchestrator, "_process_async", new_callable=AsyncMock, return_value="mocked_processing_response"
        ) as mock_process_async:
            xpia_orchestrator = XPIATestOrchestrator(
                attack_content="test",
                processing_prompt="some instructions and the required <test>",
                processing_target=processing_target,
                attack_setup_target=attack_setup_target,
                scorer=success_scorer,
            )

            score = await xpia_orchestrator.execute_async()

            assert score is not None
            assert score.get_value()
            success_scorer.score_text_async.assert_called_once()
            mock_send_to_processing_target.assert_not_called()
            mock_process_async.assert_called_once()


@pytest.mark.asyncio
async def test_xpia_orchestrator_process_async(attack_setup_target, processing_target, success_scorer):
    with patch.object(processing_target, "send_prompt_async") as mock_send_to_processing_target:
        with patch.object(
            XPIATestOrchestrator,
            "_process_async",
            new_callable=AsyncMock,
            return_value="mocked_processing_response",
        ) as mock_process_async:
            mock_send_to_processing_target.side_effect = NotImplementedError()
            xpia_orchestrator = XPIATestOrchestrator(
                attack_content="test",
                processing_prompt="some instructions and the required <test>",
                processing_target=processing_target,
                attack_setup_target=attack_setup_target,
                scorer=success_scorer,
            )
            score = await xpia_orchestrator.execute_async()
            assert score.get_value()
            success_scorer.score_text_async.assert_called_once()
            mock_send_to_processing_target.assert_not_called()
            mock_process_async.assert_awaited_once()


@pytest.mark.asyncio
@patch(
    "pyrit.common.default_values.get_non_required_value",
    return_value='{"op_name": "dummy_op"}',
)
async def test_xpia_orchestrator_process_async_with_memory_labels(
    mock_get_non_required_value, attack_setup_target, processing_target, success_scorer
):
    with patch.object(processing_target, "send_prompt_async") as mock_send_to_processing_target:
        with patch.object(
            XPIATestOrchestrator,
            "_process_async",
            new_callable=AsyncMock,
            return_value="mocked_processing_response",
        ) as mock_process_async:
            mock_send_to_processing_target.side_effect = NotImplementedError()
            xpia_orchestrator = XPIATestOrchestrator(
                attack_content="test",
                processing_prompt="some instructions and the required <test>",
                processing_target=processing_target,
                attack_setup_target=attack_setup_target,
                scorer=success_scorer,
            )
            score = await xpia_orchestrator.execute_async()
            assert score.get_value()
            entries = xpia_orchestrator.get_memory()
            assert len(entries) == 2
            assert entries[0].labels == {"op_name": "dummy_op"}

            success_scorer.score_text_async.assert_called_once()
            mock_send_to_processing_target.assert_not_called()
            mock_process_async.assert_awaited_once()
