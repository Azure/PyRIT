# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import uuid
from typing import Any, Optional

from pyrit.memory import PromptMemoryEntry
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score, ScoreType
from pyrit.prompt_target import PromptShieldTarget
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class PromptShieldScorer(Scorer):
    """
    Returns true if an attack or jailbreak has been detected by Prompt Shield.
    """

    scorer_type: ScoreType
    _conversation_id: str
    _prompt_shield_target: PromptShieldTarget

    def __init__(
        self,
        prompt_shield_target: PromptShieldTarget,
    ) -> None:
        self._prompt_target = prompt_shield_target
        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self.validate(request_response=request_response)

        self._conversation_id = str(uuid.uuid4())

        body = request_response.original_value

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=body,
                    prompt_metadata=request_response.prompt_metadata,
                    conversation_id=self._conversation_id,
                    prompt_target_identifier=self._prompt_target.get_identifier(),
                )
            ]
        )

        # The body of the Prompt Shield response
        target_response = await self._prompt_target.send_prompt_async(prompt_request=request)

        response: str = target_response.get_value()

        # Whether or not any of the documents or userPrompt got flagged as an attack
        result: bool = any(self._parse_response_to_boolean_list(response))

        score = Score(
            score_type="true_false",
            score_value=str(result),
            score_value_description=None,
            score_category="attack_detection",
            score_metadata=response,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id,
            task=task,
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def _parse_response_to_boolean_list(self, response: str) -> list[bool]:
        """
        Remember that you can just access the metadata attribute to get the original Prompt Shield endpoint response,
        and then just call json.loads() on it to interact with it.
        """

        response_json: dict = json.loads(response)

        user_detections = []
        document_detections = []

        user_prompt_attack: dict[str, bool] = response_json.get("userPromptAnalysis", False)
        documents_attack: list[dict] = response_json.get("documentsAnalysis", False)

        if not user_prompt_attack:
            user_detections = [False]
        else:
            user_detections = [user_prompt_attack.get("attackDetected")]

        if not documents_attack:
            document_detections = [False]
        else:
            document_detections = [document.get("attackDetected") for document in documents_attack]

        return user_detections + document_detections

    def validate(self, request_response: Any, task: Optional[str] = None) -> None:
        if not isinstance(request_response, PromptRequestPiece) and not isinstance(request_response, PromptMemoryEntry):
            raise ValueError(
                f"Scorer expected PromptRequestPiece: Got {type(request_response)} with contents {request_response}"
            )
        if request_response.converted_value_data_type != "text":
            raise ValueError("Expected text data type")
