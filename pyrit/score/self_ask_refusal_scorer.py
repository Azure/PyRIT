# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

from pathlib import Path
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptRequestPiece, Score, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer

REFUSAL_SCORE_SYSTEM_PROMPT = Path(DATASETS_PATH, "score", "refusal.yaml").resolve()


class SelfAskRefusalScorer(Scorer):
    """A self-ask scorer detects a refusal."""

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        memory: MemoryInterface = None,
    ) -> None:

        self.scorer_type = "true_false"

        self._prompt_target = chat_target
        self._memory = memory if memory else DuckDBMemory()

        self._system_prompt = (PromptTemplate.from_yaml_file(REFUSAL_SCORE_SYSTEM_PROMPT)).template

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """Scores the prompt and determines whether the response is a refusal.

        Args:
            request_response (PromptRequestPiece): The piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The request_response scored.
        """
        self.validate(request_response, task=task)

        conversation_id = str(uuid.uuid4())

        self._prompt_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=request_response.converted_value,
                    original_value_data_type=request_response.original_value_data_type,
                    converted_value=request_response.converted_value,
                    converted_value_data_type=request_response.converted_value_data_type,
                    conversation_id=conversation_id,
                    prompt_target_identifier=self._prompt_target.get_identifier(),
                )
            ]
        )

        score = await self.send_chat_target_async(
            prompt_target=self._prompt_target,
            scorer_llm_request=request,
            scored_prompt_id=request_response.id,
            category="refusal",
            task=task,
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        pass
