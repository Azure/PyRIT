# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
import yaml
import enum

from pathlib import Path
from typing import Dict, List, Optional, Union


from pyrit.exceptions.exception_classes import InvalidJsonException, pyrit_json_retry
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.score import Score, Scorer
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import SCALES_PATH


class ScalePaths(enum.Enum):
    TREE_OF_ATTACKS_WITH_PRUNING_SCALE = Path(SCALES_PATH, "tree_of_attacks_with_pruning.yaml").resolve()


class SelfAskScaleScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a likert scale.
    """

    def __init__(
        self,
        *, 
        chat_target: PromptChatTarget,
        scale_path: Optional[Path] = None,
        scale: Optional[Dict[str, Union[int, str]]] = None,
        memory: MemoryInterface = None) -> None:

        self.scorer_type = "float_scale"

        self._memory = memory if memory else DuckDBMemory()

        if not scale_path and not scale:
            raise ValueError("Either scale_path or scale must be provided.")
        if scale_path and scale:
            raise ValueError("Only one of scale_path or scale should be provided.")
        if scale_path:
            scale = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
        else:
            for key in ["category", "minimum_value", "minimum_description", "maximum_value", "maximum_description", "step_description"]:
                if key not in scale:
                    raise ValueError(f"{key} must be provided in scale.")

        if int(scale["minimum_value"]) >= int(scale["maximum_value"]):
            raise ValueError("minimum_value must be less than maximum_value")
        
        self._score_category = scale["category"]
        self._minimum_value = int(scale["minimum_value"])
        self._maximum_value = int(scale["maximum_value"])

        scoring_instructions_template = PromptTemplate.from_yaml_file(SCALES_PATH / "scale_system_prompt.yaml")
        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
            category=self._score_category,
            minimum_value=str(scale["minimum_value"]),
            minimum_description=scale["minimum_description"],
            maximum_value=str(scale["maximum_value"]),
            maximum_description=scale["maximum_description"],
            step_description=scale["step_description"]
        )

        self._chat_target: PromptChatTarget = chat_target

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the likert_scale.
                         The score_value is a value from [0,1] that is scaled from the likert scale.
        """
        self.validate(request_response)

        conversation_id = str(uuid.uuid4())

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

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
    async def send_chat_target_async(self, request, request_response_id):
        response = await self._chat_target.send_prompt_async(prompt_request=request)

        try:
            response_json = response.request_pieces[0].converted_value
            parsed_response = json.loads(response_json)
            score_value = self.scale_value_float(float(parsed_response["score_value"]), self._minimum_value, self._maximum_value)
            score = Score(
                score_value=str(score_value),
                score_value_description=parsed_response["description"],
                score_type=self.scorer_type,
                score_category=self._score_category,
                score_rationale=parsed_response["rationale"],
                scorer_class_identifier=self.get_identifier(),
                score_metadata=None,
                prompt_request_response_id=request_response_id,
            )
        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON response: {response_json}")

        except KeyError:
            raise InvalidJsonException(message=f"Invalid JSON response, missing Key: {response_json}")

        return score

    def validate(self, request_response: PromptRequestPiece):
        pass
