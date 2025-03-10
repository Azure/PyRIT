import pytest
import asyncio
from unittest.mock import MagicMock, patch

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.score import Score
from pyrit.score.human_in_the_loop_gradio import HumanInTheLoopScorerGradio

@pytest.fixture
def score() -> Score:
    return Score(
        score_value="true",
        score_type="true_false",
        score_category="safety",
        score_value_description="Safe",
        score_rationale="The prompt was marked safe",
        score_metadata=None,
        prompt_request_response_id="1234",
    )

@patch("pyrit.ui.rpc.AppRPCServer")
def test_scorer_start_stop_rpc_server(mock_rpc_server):
    scorer = HumanInTheLoopScorerGradio()
    mock_rpc_server.return_value.start.assert_called_once()
    scorer.__del__()
    mock_rpc_server.return_value.stop.assert_called_once()


@patch("pyrit.ui.rpc.AppRPCServer")
def test_scorer_validate(_):
    scorer = HumanInTheLoopScorerGradio()
    prompt = MagicMock()
    prompt.converted_value_data_type = "text"
    scorer.validate(prompt)
    prompt.converted_value_data_type = "image"
    with pytest.raises(ValueError):
        scorer.validate(prompt)

@patch("pyrit.ui.rpc.AppRPCServer")
def test_scorer_retrieve_score(mock_rpc_server, score: Score):
    scorer = HumanInTheLoopScorerGradio()
    prompt = MagicMock()
    prompt.converted_value_data_type = "text"

    rpc_mocked = mock_rpc_server.return_value

    rpc_mocked.wait_for_score.return_value = score

    score_result = scorer.retrieve_score(prompt)

    assert score_result[0].score_value == score.score_value

    rpc_mocked.wait_for_client.assert_called_once()
    rpc_mocked.send_score_prompt.assert_called_once_with(prompt)
    rpc_mocked.wait_for_score.assert_called_once()

@patch("pyrit.ui.rpc.AppRPCServer")
@pytest.mark.asyncio
async def test_scorer_score_async(mock_rpc_server, score: Score):
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = HumanInTheLoopScorerGradio()
        prompt = MagicMock()
        prompt.converted_value_data_type = "text"

        rpc_mocked = mock_rpc_server.return_value

        rpc_mocked.wait_for_score.return_value = score

        score_result = await scorer.score_async(prompt)

        assert score_result[0].score_value == score.score_value

        rpc_mocked.wait_for_client.assert_called_once()
        rpc_mocked.send_score_prompt.assert_called_once_with(prompt)
        rpc_mocked.wait_for_score.assert_called_once()

        memory.add_scores_to_memory.assert_called_once_with(scores=[score])

@patch("pyrit.ui.rpc.AppRPCServer")
@pytest.mark.asyncio
async def test_scorer_score_async_on_cancel_stops_server(mock_rpc_server):
    scorer = HumanInTheLoopScorerGradio()
    prompt = MagicMock()
    prompt.converted_value_data_type = "text"

    rpc_mocked = mock_rpc_server.return_value
    rpc_mocked.wait_for_score.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await scorer.score_async(prompt)

    rpc_mocked.stop.assert_called_once()
