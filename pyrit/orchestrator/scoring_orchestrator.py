# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from datetime import datetime
from typing import Optional, Sequence

from pyrit.models import PromptRequestPiece, Score
from pyrit.orchestrator import Orchestrator
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class ScoringOrchestrator(Orchestrator):
    """
    This orchestrator scores prompts in a parallelizable and convenient way.
    """

    def __init__(
        self,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If using a scorer that takes a prompt target, and providing max requests per
                minute on the target, this should be set to 1 to ensure proper rate limit management.
        """
        super().__init__(verbose=verbose)

        self._batch_size = batch_size

    async def score_prompts_by_id_async(
        self,
        *,
        scorer: Scorer,
        prompt_ids: list[str],
        responses_only: bool = False,
        task: str = "",
    ) -> list[Score]:
        """
        Scores prompts using the Scorer for prompts with the prompt_ids. Use this function if you want to score
        prompt requests as well as prompt responses, or if you want more fine-grained control over the scorer
        tasks. If you only want to score prompt responses, use the `score_responses_by_filters_async` function.

        Args:
            scorer (Scorer): The Scorer object to use for scoring.
            prompt_ids (list[str]): A list of prompt IDs correlating to the prompts to score.
            responses_only (bool, optional): If True, only the responses (messages with role "assistant") are
                scored. Defaults to False.
            task (str, optional): A task is used to give the scorer more context on what exactly to score.
                A task might be the request prompt text or the original attack model's objective.
                **Note: the same task is to applied to all prompt_ids.** Defaults to an empty string.

        Returns:
            list[Score]: A list of Score objects for the prompts with the prompt_ids.
        """
        request_pieces: Sequence[PromptRequestPiece] = []
        request_pieces = self._memory.get_prompt_request_pieces(prompt_ids=prompt_ids)

        if responses_only:
            request_pieces = self._extract_responses_only(request_pieces)

        request_pieces = self._remove_duplicates(request_pieces)

        return await scorer.score_prompts_with_tasks_batch_async(
            request_responses=request_pieces, batch_size=self._batch_size, tasks=[task] * len(request_pieces)
        )

    async def score_responses_by_filters_async(
        self,
        *,
        scorer: Scorer,
        orchestrator_id: Optional[str | uuid.UUID] = None,
        conversation_id: Optional[str | uuid.UUID] = None,
        prompt_ids: Optional[list[str] | list[uuid.UUID]] = None,
        labels: Optional[dict[str, str]] = None,
        sent_after: Optional[datetime] = None,
        sent_before: Optional[datetime] = None,
        original_values: Optional[list[str]] = None,
        converted_values: Optional[list[str]] = None,
        data_type: Optional[str] = None,
        not_data_type: Optional[str] = None,
        converted_value_sha256: Optional[list[str]] = None,
    ) -> list[Score]:
        """
        Scores the responses that match the specified filters.

        Args:
            scorer (Scorer): The Scorer object to use for scoring.
            orchestrator_id (Optional[str | uuid.UUID], optional): The ID of the orchestrator. Defaults to None.
            conversation_id (Optional[str | uuid.UUID], optional): The ID of the conversation. Defaults to None.
            prompt_ids (Optional[list[str] | list[uuid.UUID]], optional): A list of prompt IDs. Defaults to None.
            labels (Optional[dict[str, str]], optional): A dictionary of labels. Defaults to None.
            sent_after (Optional[datetime], optional): Filter for prompts sent after this datetime. Defaults to None.
            sent_before (Optional[datetime], optional): Filter for prompts sent before this datetime. Defaults to None.
            original_values (Optional[list[str]], optional): A list of original values. Defaults to None.
            converted_values (Optional[list[str]], optional): A list of converted values. Defaults to None.
            data_type (Optional[str], optional): The data type to filter by. Defaults to None.
            not_data_type (Optional[str], optional): The data type to exclude. Defaults to None.
            converted_value_sha256 (Optional[list[str]], optional): A list of SHA256 hashes of converted values.
                Defaults to None.
        Returns:
            list[Score]: A list of Score objects for responses that match the specified filters.
        Raises:
            Exception: If there is an error retrieving the prompts,
                an exception is logged and an empty list is returned.
        """
        request_pieces: Sequence[PromptRequestPiece] = []
        request_pieces = self._memory.get_prompt_request_pieces(
            orchestrator_id=orchestrator_id,
            conversation_id=conversation_id,
            prompt_ids=prompt_ids,
            labels=labels,
            sent_after=sent_after,
            sent_before=sent_before,
            original_values=original_values,
            converted_values=converted_values,
            data_type=data_type,
            not_data_type=not_data_type,
            converted_value_sha256=converted_value_sha256,
        )

        request_pieces = self._remove_duplicates(request_pieces)

        if not request_pieces:
            raise ValueError("No entries match the provided filters. Please check your filters.")

        return await scorer.score_responses_inferring_tasks_batch_async(
            request_responses=request_pieces, batch_size=self._batch_size
        )

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
