# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib.util
import time
from threading import Event, Thread
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score
from pyrit.score import HumanInTheLoopScorerGradio
from pyrit.ui.rpc import RPCAlreadyRunningException
from pyrit.ui.rpc_client import RPCClient, RPCClientStoppedException


def if_gradio_installed():
    return importlib.util.find_spec("gradio") is not None


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


@pytest.fixture
def promptOriginal() -> PromptRequestPiece:
    return PromptRequestPiece(
        role="assistant",
        original_value="This is the original value",
        converted_value="This is the converted value",
    )


class IntegrationRpcClient:
    def __init__(self, score_callback: Callable[[PromptRequestPiece], bool], disconnect_callback: Optional[Callable]):
        self._score_callback = score_callback
        self.rpc_client = RPCClient(disconnect_callback)
        self._is_running = False
        self._thread = None  # type: Optional[Thread]
        self._thread_exception = None  # type: Optional[Exception]

    def start(self):
        self._is_running = True
        self._thread = Thread(target=self._run)
        self._thread.start()

    def _run(self):
        self.rpc_client.start()
        try:
            while self._is_running:
                prompt = self.rpc_client.wait_for_prompt()
                response = self._score_callback(prompt)
                self.rpc_client.send_prompt_response(response)
        except RPCClientStoppedException as e:
            if self._is_running:
                self._thread_exception = e
        except Exception as e:
            self._thread_exception = e
        finally:
            if self._is_running:
                self.rpc_client.stop()

    def stop(self):
        self._is_running = False
        self.rpc_client.stop()
        self._thread.join()

        if self._thread_exception is not None:
            raise self._thread_exception


@pytest.mark.skipif(not if_gradio_installed(), reason="Gradio is not installed")
class TestHiTLGradioIntegration:

    @patch("pyrit.ui.rpc.is_app_running")
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_scorer_can_start(self, mock_is_app_running, promptOriginal: PromptRequestPiece):
        memory = MagicMock(MemoryInterface)
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

            disconnected_event = Event()

            def disconnected():
                print("Disconnected called")
                disconnected_event.set()

            def score_callback(prompt: PromptRequestPiece) -> bool:
                assert prompt.original_value == promptOriginal.original_value
                assert prompt.converted_value == promptOriginal.converted_value
                return True

            mock_is_app_running.return_value = True

            rpc_client = IntegrationRpcClient(score_callback, disconnected)
            scorer = HumanInTheLoopScorerGradio()

            rpc_client.start()
            score_result = await scorer.score_async(promptOriginal)

            assert score_result[0].score_value == "True"
            rpc_client.stop()
            scorer.__del__()

            disconnected_event.wait(15)
            assert disconnected_event.is_set()

            mock_is_app_running.assert_called_once()

    @patch("pyrit.ui.rpc.is_app_running")
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_scorer_receive_multiple(self, mock_is_app_running, promptOriginal: PromptRequestPiece):
        memory = MagicMock(MemoryInterface)
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

            disconnected_event = Event()

            def disconnected():
                print("Disconnected called")
                disconnected_event.set()

            i = -1

            def score_callback(prompt: PromptRequestPiece) -> bool:
                nonlocal i
                i += 1
                assert prompt.original_value == promptOriginal.original_value
                assert prompt.converted_value == promptOriginal.converted_value

                if i % 2 == 0:
                    return True
                return False

            mock_is_app_running.return_value = True

            rpc_client = IntegrationRpcClient(score_callback, disconnected)
            scorer = HumanInTheLoopScorerGradio()

            rpc_client.start()

            score_result = await scorer.score_async(promptOriginal)
            assert score_result[0].score_value == "True"

            # Next prompt
            score_result = await scorer.score_async(promptOriginal)
            assert score_result[0].score_value == "False"

            score_result = await scorer.score_async(promptOriginal)
            assert score_result[0].score_value == "True"

            rpc_client.stop()
            scorer.__del__()

            disconnected_event.wait(15)
            assert disconnected_event.is_set()

            mock_is_app_running.assert_called_once()

    @patch("pyrit.ui.rpc.is_app_running")
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_scorer_handle_client_disconnect(self, mock_is_app_running):
        memory = MagicMock(MemoryInterface)
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

            disconnected_event = Event()

            def disconnected():
                print("Disconnected called")
                disconnected_event.set()

            def score_callback(prompt: PromptRequestPiece) -> bool:
                pytest.fail("Should not be called")
                return True

            mock_is_app_running.return_value = True

            rpc_client = IntegrationRpcClient(score_callback, disconnected)
            scorer = HumanInTheLoopScorerGradio()

            rpc_client.start()
            time.sleep(2)  # Wait for the client to start
            rpc_client.stop()

            scorer.__del__()

            disconnected_event.wait(15)
            assert disconnected_event.is_set()

    @patch("pyrit.ui.rpc.is_app_running")
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_scorer_handle_server_disconnect(self, mock_is_app_running):
        memory = MagicMock(MemoryInterface)
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

            disconnected_event = Event()

            def disconnected():
                print("Disconnected called")
                disconnected_event.set()

            def score_callback(prompt: PromptRequestPiece) -> bool:
                pytest.fail("Should not be called")
                return True

            mock_is_app_running.return_value = True

            rpc_client = IntegrationRpcClient(score_callback, disconnected)
            scorer = HumanInTheLoopScorerGradio()
            rpc_client.start()
            time.sleep(2)  # Wait for the client to start
            scorer.__del__()

            time.sleep(2)  # Wait for the client to stop
            rpc_client.stop()

            disconnected_event.wait(15)
            assert disconnected_event.is_set()

    @patch("pyrit.ui.rpc.is_app_running")
    @pytest.mark.timeout(30)
    def test_scorer_handle_exception_scorer(self, mock_is_app_running):
        memory = MagicMock(MemoryInterface)
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            mock_is_app_running.return_value = True

            _ = HumanInTheLoopScorerGradio()
            with pytest.raises(RPCAlreadyRunningException):
                HumanInTheLoopScorerGradio()
