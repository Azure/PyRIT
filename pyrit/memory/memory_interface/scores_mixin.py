# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scores mixin for MemoryInterface containing score-related operations."""

import logging
from typing import Sequence

from pyrit.memory.memory_models import ScoreEntry
from pyrit.models import Score

logger = logging.getLogger(__name__)


class MemoryScoresMixin:
    """Mixin providing score-related methods for memory management."""

    def add_scores_to_memory(self, *, scores: Sequence[Score]) -> None:
        """
        Inserts a list of scores into the memory storage.
        """
        for score in scores:
            if score.prompt_request_response_id:
                prompt_request_response_id = score.prompt_request_response_id
                prompt_piece = self.get_prompt_request_pieces(prompt_ids=[str(prompt_request_response_id)])
                if not prompt_piece:
                    logging.error(f"Prompt with ID {prompt_request_response_id} not found in memory.")
                    continue
                # auto-link score to the original prompt id if the prompt is a duplicate
                if prompt_piece[0].original_prompt_id != prompt_piece[0].id:
                    score.prompt_request_response_id = prompt_piece[0].original_prompt_id
        self._insert_entries(entries=[ScoreEntry(entry=score) for score in scores])

    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: Sequence[str]) -> Sequence[Score]:
        """
        Gets a list of scores based on prompt_request_response_ids.
        """
        prompt_pieces = self.get_prompt_request_pieces(prompt_ids=prompt_request_response_ids)
        # Get the original prompt IDs from the prompt pieces so correct scores can be obtained
        prompt_request_response_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        entries: Sequence[ScoreEntry] = self._query_entries(
            ScoreEntry, conditions=ScoreEntry.prompt_request_response_id.in_(prompt_request_response_ids)
        )

        return [entry.get_score() for entry in entries]

    def get_scores_by_orchestrator_id(self, *, orchestrator_id: str) -> Sequence[Score]:
        """
        Retrieves a list of Score objects associated with the PromptRequestPiece objects
        which have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            Sequence[Score]: A list of Score objects associated with the PromptRequestPiece objects
                which match the specified orchestrator ID.
        """
        prompt_pieces = self.get_prompt_request_pieces(orchestrator_id=orchestrator_id)
        # Since duplicate pieces do not have their own score entries, get the original prompt IDs from the pieces.
        prompt_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        return self.get_scores_by_prompt_ids(prompt_request_response_ids=prompt_ids)

    def get_scores_by_memory_labels(self, *, memory_labels: dict[str, str]) -> Sequence[Score]:
        """
        Retrieves a list of Score objects associated with the PromptRequestPiece objects
        which have the specified memory labels.

        Args:
            memory_labels (dict[str, str]): A free-form dictionary for tagging prompts with custom labels.
                These labels can be used to track all prompts sent as part of an operation, score prompts based on
                the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
                Users can define any key-value pairs according to their needs.

        Returns:
            Sequence[Score]: A list of Score objects associated with the PromptRequestPiece objects
                which match the specified memory labels.
        """
        prompt_pieces = self.get_prompt_request_pieces(labels=memory_labels)
        # Since duplicate pieces do not have their own score entries, get the original prompt IDs from the pieces.
        prompt_ids = [str(piece.original_prompt_id) for piece in prompt_pieces]
        return self.get_scores_by_prompt_ids(prompt_request_response_ids=prompt_ids)
