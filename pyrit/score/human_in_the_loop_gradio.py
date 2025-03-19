# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class HumanInTheLoopScorerGradio(Scorer):
    """
    Create scores from manual human input using Gradio and adds them to the database.

    Parameters:
        open_browser(bool): The scorer will open the Gradio interface in a browser instead of opening it in PyWebview
    """

    def __init__(self, *, open_browser=False) -> None:
        # Import here to avoid importing rpyc in the main module that might not be installed
        from pyrit.ui.rpc import AppRPCServer

        self._rpc_server = AppRPCServer(open_browser=open_browser)
        self._rpc_server.start()

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self.validate(request_response=request_response)
        try:
            score = await asyncio.to_thread(self.retrieve_score, request_response, task=task)
            self._memory.add_scores_to_memory(scores=score)
            return score
        except asyncio.CancelledError:
            self._rpc_server.stop()
            raise

    def retrieve_score(self, request_prompt: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self._rpc_server.wait_for_client()
        self._rpc_server.send_score_prompt(request_prompt)
        score = self._rpc_server.wait_for_score()
        score.scorer_class_identifier = self.get_identifier()
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Prompt data type must be 'text' for Gradio manual scoring.")

    def __del__(self):
        self._rpc_server.stop()
