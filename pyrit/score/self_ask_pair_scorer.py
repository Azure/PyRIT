# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from pathlib import Path

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions.exception_classes import InvalidJsonException, pyrit_json_retry
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer


class SelfAskPAIRScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a likert scale.
    """

    def __init__(
        self,
        chat_target: PromptChatTarget,
        attack_objective: str,
        memory: MemoryInterface = None,
        judge_system_prompt_template_path: Path = DATASETS_PATH
        / "score"
        / "likert_scales"
        / "judge_system_prompt.yaml",
    ) -> None:

        self.scorer_type = "float_scale"

        self._memory = memory if memory else DuckDBMemory()
        self._attack_objective = attack_objective
        judge_prompt_template = PromptTemplate.from_yaml_file(judge_system_prompt_template_path)
        self._system_prompt = judge_prompt_template.apply_custom_metaprompt_parameters(goal=self._attack_objective)
        self._chat_target: PromptChatTarget = chat_target

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.

        Returns:
            list[Score]: The request_response scored. The category is configured from the likert_scale.
        """
        self.validate(request_response)

        conversation_id = str(uuid.uuid4())

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )
        logging.info(f"Conversation ID: {conversation_id}, Request Response: {request_response.converted_value}")
        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=request_response.converted_value,
                    conversation_id=conversation_id,
                    prompt_target_identifier=self._chat_target.get_identifier(),
                )
            ]
        )

        score = await self.send_chat_target_async(request, request_response.id)
        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    @pyrit_json_retry
    async def send_chat_target_async(self, request: PromptRequestResponse, request_response_id: str) -> Score:
        response = await self._chat_target.send_prompt_async(prompt_request=request)
        score_number = response.request_pieces[0].converted_value

        try:
            score_number_float = float(score_number)
        except (ValueError, TypeError):
            # Raising `InvalidJsonException` because it's the only error class the current @pyrit_json_retry decorator
            # will catch and handle.
            raise InvalidJsonException(message=f"Malformed response cannot be converted to integer: {score_number}")
        # Normalize the score to a 0-1 scale. The PAIR Likert scale is 1-10.
        normalized_score_value = self.scale_value_float(value=score_number_float, min_value=1.0, max_value=10.0)
        score = Score(
            score_value=str(normalized_score_value),
            score_value_description="PAIR Likert Scale Score. The scale is 1-10 scale but normalized to 0.0 to 1.0",
            score_type=self.scorer_type,
            score_category="PAIR",
            score_rationale="",
            scorer_class_identifier=self.get_identifier(),
            score_metadata="",
            prompt_request_response_id=request_response_id,
        )

        return score

    def validate(self, request_response: PromptRequestPiece):
        pass
