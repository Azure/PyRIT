# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal

from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.memory import CentralMemory
from pyrit.models import Score

logger = logging.getLogger(__name__)


OrchestratorResultStatus = Literal["success", "failure", "pruned", "adversarial_generation", "in_progress", "error", "unknown"]


class OrchestratorResult:
    """The result of an orchestrator."""

    def __init__(
        self,
        conversation_id: str,
        objective: str,
        status: OrchestratorResultStatus = "in_progress",
        score: Score = None,
        confidence: float = 0.1,
    ):
        self.conversation_id = conversation_id
        self.objective = objective
        self.status = status
        self.score = score
        self.confidence = confidence

        self._memory = CentralMemory.get_memory_instance()

    async def print_conversation_async(self):
        """Prints the conversation between the objective target and the adversarial chat, including the scores.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
        """
        target_messages = self._memory.get_conversation(conversation_id=self.conversation_id)

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        if self.status == "success":
            print(
                f"{Style.BRIGHT}{Fore.RED}The orchestrator has completed the conversation and achieved "
                f"the objective: {self.objective}"
            )
        elif self.status == "failure":
            print(
                f"{Style.BRIGHT}{Fore.RED}The orchestrator has not achieved the objective: "
                f"{self.objective}"
            )
        else:
            print(
                f"{Style.BRIGHT}{Fore.RED}The orchestrator with objective: {self.objective} "
                f"has ended with status: {self.status}"
            )

        if self.score:
            print(f"{Style.BRIGHT}{Fore.RED}objective score: {self.score} : {self.score.score_rationale}")

        for message in target_messages:
            for piece in message.request_pieces:
                if piece.role == "user":
                    print(f"{Style.BRIGHT}{Fore.BLUE}{piece.role}:")
                    if piece.converted_value != piece.original_value:
                        print(f"Original value: {piece.original_value}")
                    print(f"Converted value: {piece.converted_value}")
                else:
                    print(f"{Style.NORMAL}{Fore.YELLOW}{piece.role}: {piece.converted_value}")

                await display_image_response(piece)

                auxiliary_scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(piece.id)])
                if auxiliary_scores and len(auxiliary_scores) > 0:
                    for auxiliary_score in auxiliary_scores:
                        if (not self.score) or (auxiliary_score.id != self.score.id):
                            print(f"{Style.RESET_ALL}auxiliary score: {auxiliary_score} : {auxiliary_score.score_rationale}")
