# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Sequence

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.models import Score
from pyrit.orchestrator import Orchestrator
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class ScoringOrchestrator(Orchestrator):
    """
    This orchestrator scores prompts in a parallelizable and convenient way.
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
                Note: If using a scorer that takes a prompt target, and providing max requests per
                minute on the target, this should be set to 1 to ensure proper rate limit management.
        """
        super().__init__(memory=memory, verbose=verbose)

        self._batch_size = batch_size

    def _assign_memory(self, scorer: Scorer) -> None:
        """
        Assigns the provided memory to the scorer passed to the orchestrator,
        as well as to its prompt target, if applicable.
        """
        # Set the scorer and scorer._prompt_target memory to match the orchestrator's memory.
        scorer._memory = self._memory
        if hasattr(scorer, "_prompt_target"):
            scorer._prompt_target._memory = self._memory

    async def score_prompts_by_orchestrator_id_async(
        self,
        *,
        scorer: Scorer,
        orchestrator_ids: list[str],
        responses_only: bool = True,
    ) -> list[Score]:
        """
        Scores prompts using the Scorer for prompts correlated to the orchestrator_ids.
        """
        self._assign_memory(scorer)
        request_pieces: list[PromptRequestPiece] = []
        for id in orchestrator_ids:
            request_pieces.extend(self._memory.get_prompt_request_piece_by_orchestrator_id(orchestrator_id=id))
        if responses_only:
            request_pieces = self._extract_responses_only(request_pieces)
        request_pieces = self._remove_duplicates(request_pieces)

        return await scorer.score_prompts_batch_async(request_responses=request_pieces, batch_size=self._batch_size)

    async def score_prompts_by_memory_labels_async(
        self,
        *,
        scorer: Scorer,
        memory_labels: dict[str, str] = {},
        responses_only: bool = True,
    ) -> list[Score]:
        """
        Scores prompts using the Scorer for prompts based on the memory labels.
        """
        if not memory_labels:
            raise ValueError("Invalid memory_labels: Please provide valid memory labels.")
        self._assign_memory(scorer)
        request_pieces: list[PromptRequestPiece] = self._memory.get_prompt_request_piece_by_memory_labels(
            memory_labels=memory_labels
        )
        if not request_pieces:
            raise ValueError("No entries match the provided memory labels. Please check your memory labels.")

        if responses_only:
            request_pieces = self._extract_responses_only(request_pieces)

        request_pieces = self._remove_duplicates(request_pieces)

        return await scorer.score_prompts_batch_async(request_responses=request_pieces, batch_size=self._batch_size)

    async def score_prompts_by_request_id_async(
        self,
        *,
        scorer: Scorer,
        prompt_ids: list[str],
        responses_only: bool = False,
    ) -> list[Score]:
        """
        Scores prompts using the Scorer for prompts with the prompt_ids
        """
        self._assign_memory(scorer)
        request_pieces: Sequence[PromptRequestPiece] = []
        request_pieces = self._memory.get_prompt_request_pieces_by_id(prompt_ids=prompt_ids)

        if responses_only:
            request_pieces = self._extract_responses_only(request_pieces)

        request_pieces = self._remove_duplicates(request_pieces)

        return await scorer.score_prompts_batch_async(request_responses=request_pieces, batch_size=self._batch_size)

    def _extract_responses_only(self, request_responses: Sequence[PromptRequestPiece]) -> list[PromptRequestPiece]:
        """
        Extracts the responses from the list of PromptRequestPiece objects.
        """
        return [response for response in request_responses if response.role == "assistant"]

    def _remove_duplicates(self, request_responses: Sequence[PromptRequestPiece]) -> list[PromptRequestPiece]:
        """
        Removes the duplicates from the list of PromptRequestPiece objects so that identical prompts are not
        scored twice.
        """
        return [response for response in request_responses if response.original_prompt_id == response.id]
