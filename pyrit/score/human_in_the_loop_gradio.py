import asyncio
from pyrit.score.scorer import Scorer
from pyrit.models import Score, PromptRequestPiece
from typing import Optional

from ui.rpc import AppRpcServer


class HumanInTheLoopScorerGradio(Scorer):

    def __init__(self, *, scorer: Scorer = None, re_scorers: list[Scorer] = None) -> None:
        self._scorer = scorer
        self._re_scorers = re_scorers
        self._rpc_server = AppRpcServer()
        self._rpc_server.start()


    async def score_async(self, request: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        try:
            return await asyncio.to_thread(self.score_prompt_manually, request, task=task)
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