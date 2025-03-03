# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from pyrit.score.scorer import Scorer
from pyrit.models import Score, PromptRequestPiece
from typing import Optional

class HumanInTheLoopScorerGradio(Scorer):

    def __init__(self, *, open_browser=False, scorer: Scorer = None, re_scorers: list[Scorer] = None) -> None:
        # Import here to avoid importing rpyc in the main module that might not be installed
        from pyrit.ui.rpc import AppRpcServer

        self._scorer = scorer
        self._re_scorers = re_scorers
        self._rpc_server = AppRpcServer(open_browser=open_browser)
        self._rpc_server.start()


    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        try:
            return await asyncio.to_thread(self.score_prompt_manually, request_response, task=task)
        except asyncio.CancelledError:
            self._rpc_server.stop()
            raise
        
    
    def score_prompt_manually(self, request_prompt: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self._rpc_server.wait_for_client()
        self._rpc_server.send_score_prompt(request_prompt)
        score = self._rpc_server.wait_for_score()
        return [score]
    
    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass

    def __del__(self):
        self._rpc_server.stop()