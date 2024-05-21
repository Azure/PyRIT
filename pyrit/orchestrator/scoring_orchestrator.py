# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class ScoringOrchestrator(Orchestrator):
    """
    This orchestrator scores prompts in a parallizable and convenient way.
    """

    def __init__(
        self,
        memory: MemoryInterface = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
        """
        super().__init__(memory=memory, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._batch_size = batch_size

    async def score_prompts_by_orchestrator_id_async(
        self,
        *,
        scorer: Scorer,
        orchestrator_ids: list[int],
        responses_only: bool = True,
    ) -> list[Score]:
        """
        Scores prompts using the Scorer for prompts correlated to the orchestrator_ids.
        """

        request_pieces: list[PromptRequestPiece] = []
        for id in orchestrator_ids:
            request_pieces.extend(self._memory.get_prompt_request_piece_by_orchestrator_id(orchestrator_id=int(id)))

        if responses_only:
            request_pieces = self._extract_responses_only(request_pieces)

        return await self._score_prompts_batch_async(prompts=request_pieces, scorer=scorer)

    async def score_prompts_by_request_id_async(self, *, scorer: Scorer, prompt_ids: list[str]) -> list[Score]:
        """
        Scores prompts using the Scorer for prompts with the prompt_ids
        """

        requests: list[PromptRequestPiece] = []
        requests = self._memory.get_prompt_request_pieces_by_id(prompt_ids=prompt_ids)

        return await self._score_prompts_batch_async(prompts=requests, scorer=scorer)

    def _extract_responses_only(self, request_responses: list[PromptRequestPiece]) -> list[PromptRequestPiece]:
        """
        Extracts the responses from the list of PromptRequestResponse objects.
        """
        return [response for response in request_responses if response.role == "assistant"]

    async def _score_prompts_batch_async(self, prompts: list[PromptRequestPiece], scorer: Scorer) -> list[Score]:
        results = []

        for prompts_batch in self._chunked_prompts(prompts, self._batch_size):
            tasks = []
            for prompt in prompts_batch:
                tasks.append(scorer.score_async(request_response=prompt))

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        # results is a list[list[str]] and needs to be flattened
        return [score for sublist in results for score in sublist]

    def _chunked_prompts(self, prompts, size):
        for i in range(0, len(prompts), size):
            yield prompts[i : i + size]
