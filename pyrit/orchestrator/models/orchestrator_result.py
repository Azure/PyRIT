# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Annotated, Literal

from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.memory import CentralMemory
from pyrit.models import Score

logger = logging.getLogger(__name__)


OrchestratorResultStatus = Annotated[
    Literal["success", "failure", "pruned", "adversarial_generation", "in_progress", "error", "unknown"],
    """The status of an orchestrator result.

    Completion States:
        - success: The orchestrator run is complete and achieved its objective.
        - failure: The orchestrator run is complete and failed to achieve its objective.
        - error: The orchestrator run is complete and encountered an error.
        - unknown: The orchestrator run is complete and it is unknown whether it achieved its objective.

    Intermediate States:
        - in_progress: The orchestrator is still running.

    Special States:
        - pruned: The conversation was pruned as part of an attack and not related to success/failure/unknown/error.
        - adversarial_generation: The conversation was used as part of adversarial generation and not related to
          success/failure/unknown/error.
    """,
]


class OrchestratorResult:
    """The result of an orchestrator."""

    def __init__(
        self,
        conversation_id: str,
        objective: str,
        status: OrchestratorResultStatus = "in_progress",
        objective_score: Score = None,
        confidence: float = 0.1,
    ):
        self.conversation_id = conversation_id
        self.objective = objective
        self.status = status
        self.objective_score = objective_score
        self.confidence = confidence

        self._memory = CentralMemory.get_memory_instance()

    async def print_conversation_async(self, include_auxiliary_scores: bool = False):
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
            print(f"{Style.BRIGHT}{Fore.RED}The orchestrator has not achieved the objective: " f"{self.objective}")
        else:
            print(
                f"{Style.BRIGHT}{Fore.RED}The orchestrator with objective: {self.objective} "
                f"has ended with status: {self.status}"
            )

        for message in target_messages:
            for piece in message.request_pieces:
                if piece.role == "user":
                    if piece.converted_value != piece.original_value:
                        print(f"\nOriginal value: {piece.original_value}")
                    print(f"\n{Style.BRIGHT}{Fore.LIGHTBLACK_EX}{piece.role.capitalize()}: {Style.NORMAL}{piece.converted_value}")
                else:
                    print(f"{Style.BRIGHT}{Fore.LIGHTBLACK_EX}{piece.role.capitalize()}: {Style.NORMAL}{piece.converted_value}")

                await display_image_response(piece)

                if include_auxiliary_scores:
                    auxiliary_scores = (
                        self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(piece.id)]) or []
                    )
                    for auxiliary_score in auxiliary_scores:
                        if not self.objective_score or auxiliary_score.id != self.objective_score.id:
                            print(
                                f"{Style.DIM}{Fore.WHITE}auxiliary score: {auxiliary_score} : "
                                f"{auxiliary_score.score_rationale}"
                            )

        if self.objective_score:
            print(
                f"{Style.NORMAL}{Fore.WHITE}objective score: {self.objective_score} : "
                f"{self.objective_score.score_rationale}"
            )


    async def get_conversation_report_async(self) -> dict:
        """
        Returns a structured conversation report for HTML reporting.

        Groups user and assistant messages into turns:
          - Each turn contains up to two messages: one user message, one assistant message.
          - Each piece stores both original_value and converted_value.
          - Assistant pieces may have scores if found in memory.
          - The final score is taken from the last assistant piece with a score in the transcript.
        """
        report = {}
        transcript = []
        scores_by_turn = []

        # Retrieve conversation messages from memory using the conversation ID.
        target_messages = self._memory.get_conversation(conversation_id=self.conversation_id)
        if not target_messages:
            report["error"] = "No conversation with the target"
            return report

        # Set basic conversation info.
        report["objective"] = self.objective
        report["achieved_objective"] = self.status == "success"

        turn_index = 1
        i = 0
        n = len(target_messages)

        # Process messages in pairs: (user, assistant).
        while i < n:
            turn_data = {"turn_index": turn_index, "pieces": []}

            # 1) Process user message (if exists)
            if i < n:
                turn_data["pieces"].extend(
                    self._build_piece_data(
                        target_messages[i],
                        turn_index=turn_index,
                        scores_by_turn=scores_by_turn,
                        is_assistant=False
                    )
                )
                i += 1

            # 2) Process assistant message (if exists)
            if i < n:
                turn_data["pieces"].extend(
                    self._build_piece_data(
                        target_messages[i],
                        turn_index=turn_index,
                        scores_by_turn=scores_by_turn,
                        is_assistant=True
                    )
                )
                i += 1

            transcript.append(turn_data)
            turn_index += 1

        # Determine the final score from the last assistant piece that has scores.
        final_score = None
        for turn in reversed(transcript):
            for piece in reversed(turn["pieces"]):
                if piece["role"].lower() == "assistant" and piece["scores"]:
                    final_score = piece["scores"][0]["score"]
                    break
            if final_score is not None:
                break

        report["transcript"] = transcript
        report["aggregated_metrics"] = {
            "total_turns": len(transcript),
            "final_score": final_score,
            "scores_by_turn": scores_by_turn
        }
        report["additional_metadata"] = {
            "conversation_id": self.conversation_id
        }

        return report

    def _build_piece_data(
            self,
            message,
            turn_index: int,
            scores_by_turn: list,
            is_assistant: bool
    ) -> list:
        """
        Helper function to convert each piece in a message into a dict with:
          - role
          - original_value
          - converted_value
          - scores (assistant only)

        If 'is_assistant' is True, we retrieve scores from memory.
        If any scores are found, they are added to both the piece data
        and 'scores_by_turn' for step-by-step breakdown.
        """
        pieces_data = []
        for piece in message.request_pieces:
            piece_data = {
                "role": piece.role,
                "original_value": piece.original_value or "",
                "converted_value": piece.converted_value or "",
                "scores": []
            }

            # Only retrieve scores if this piece is truly from the assistant
            if is_assistant and piece.role.lower() == "assistant":
                scores = self._memory.get_scores_by_prompt_ids(
                    prompt_request_response_ids=[str(piece.id)]
                )
                if scores:
                    piece_scores = []
                    for s in scores:
                        piece_scores.append({
                            "score": s.score_value,
                            "rationale": s.score_rationale
                        })
                    piece_data["scores"] = piece_scores

                    # Record for step-by-step breakdown
                    if piece_scores:
                        scores_by_turn.append({
                            "turn_index": turn_index,
                            "score_details": piece_scores
                        })

            pieces_data.append(piece_data)
        return pieces_data

