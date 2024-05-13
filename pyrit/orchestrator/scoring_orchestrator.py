# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from typing import Optional

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptDataType, PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
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
        self, *, scorer: Scorer, orchestrator_ids: list[str]
    ) -> list[Score]:
        """
        Sends the prompts to the prompt target.
        """

        requests: list[PromptRequestPiece] = []
        for id in orchestrator_ids:
            requests.append(self._memory.get_orchestrator_conversations(orchestrator_id=id))

        return await self._score_prompts_async(prompts=requests, scorer=scorer)



    async def _score_prompts_async(self, prompts: list[PromptRequestPiece], scorer: Scorer) -> list[Score]:
        results = []

        for prompts_batch in self._chunked_prompts(prompts, self._batch_size):
            tasks = []
            for prompt in prompts_batch:
                tasks.append(
                    scorer.score_async(request=prompt)
                )

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def _chunked_prompts(self, prompts, size):
        for i in range(0, len(prompts), size):
            yield prompts[i : i + size]

