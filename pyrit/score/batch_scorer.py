# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from datetime import datetime
from typing import Optional, Sequence

from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class BatchScorer:
    """
    A utility class for scoring prompts in batches in a parallelizable and convenient way.

    This class provides functionality to score existing prompts stored in memory
    without any target interaction, making it a pure scoring utility.
    """

    def __init__(
        self,
        *,
        batch_size: int = 10,
    ) -> None:
        """
        Initialize the BatchScorer.

        Args:
            batch_size (int): The (max) batch size for sending prompts. Defaults to 10.
                Note: If using a scorer that takes a prompt target, and providing max requests per
                minute on the target, this should be set to 1 to ensure proper rate limit management.
        """
        self._memory = CentralMemory.get_memory_instance()
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
        Score prompts using the Scorer for prompts with the prompt_ids.

        Use this function if you want to score prompt requests as well as prompt responses,
        or if you want more fine-grained control over the scorer tasks. If you only want to
        score prompt responses, use the `score_responses_by_filters_async` function.

        Args:
            scorer (Scorer): The Scorer object to use for scoring.
            prompt_ids (list[str]): A list of prompt IDs correlating to the prompts to score.
            responses_only (bool): If True, only the responses (messages with role "assistant") are
                scored. Defaults to False.
            task (str): A task is used to give the scorer more context on what exactly to score.
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
        attack_id: Optional[str | uuid.UUID] = None,
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
        Score the responses that match the specified filters.

        Args:
            scorer (Scorer): The Scorer object to use for scoring.
            attack_id (Optional[str | uuid.UUID]): The ID of the attack. Defaults to None.
            conversation_id (Optional[str | uuid.UUID]): The ID of the conversation. Defaults to None.
            prompt_ids (Optional[list[str] | list[uuid.UUID]]): A list of prompt IDs. Defaults to None.
            labels (Optional[dict[str, str]]): A dictionary of labels. Defaults to None.
            sent_after (Optional[datetime]): Filter for prompts sent after this datetime. Defaults to None.
            sent_before (Optional[datetime]): Filter for prompts sent before this datetime. Defaults to None.
            original_values (Optional[list[str]]): A list of original values. Defaults to None.
            converted_values (Optional[list[str]]): A list of converted values. Defaults to None.
            data_type (Optional[str]): The data type to filter by. Defaults to None.
            not_data_type (Optional[str]): The data type to exclude. Defaults to None.
            converted_value_sha256 (Optional[list[str]]): A list of SHA256 hashes of converted values.
                Defaults to None.

        Returns:
            list[Score]: A list of Score objects for responses that match the specified filters.

        Raises:
            ValueError: If no entries match the provided filters.
        """
        request_pieces: Sequence[PromptRequestPiece] = []
        request_pieces = self._memory.get_prompt_request_pieces(
            attack_id=attack_id,
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
        Extract the responses from the list of PromptRequestPiece objects.

        Args:
            request_responses (Sequence[PromptRequestPiece]): The request responses to filter.

        Returns:
            list[PromptRequestPiece]: A list containing only assistant responses.
        """
        return [response for response in request_responses if response.role == "assistant"]

    def _remove_duplicates(self, request_responses: Sequence[PromptRequestPiece]) -> list[PromptRequestPiece]:
        """
        Remove duplicates from the list of PromptRequestPiece objects.

        This ensures that identical prompts are not scored twice by filtering
        to only include original prompts (where original_prompt_id == id).

        Args:
            request_responses (Sequence[PromptRequestPiece]): The request responses to deduplicate.

        Returns:
            list[PromptRequestPiece]: A list with duplicates removed.
        """
        return [response for response in request_responses if response.original_prompt_id == response.id]
