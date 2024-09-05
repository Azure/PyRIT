# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
import yaml
import enum

from pathlib import Path
from typing import Optional

from pyrit.exceptions.exception_classes import InvalidJsonException, pyrit_json_retry
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.score import Score, Scorer
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import SCALES_PATH


class ScalePaths(enum.Enum):
    TREE_OF_ATTACKS_WITH_PRUNING_SCALE = Path(SCALES_PATH, "tree_of_attacks_with_pruning_jailbreak.yaml").resolve()


class ScaleExample:
    def __init__(self, *, task: str, response: str, rationale: str, score_value: int) -> None:
        self.task = task
        self.response = response
        self.rationale = rationale
        self.score_value = score_value


class Scale:
    def __init__(
        self,
        *,
        category: str,
        minimum_value: int,
        minimum_description: str,
        maximum_value: int,
        maximum_description: str,
        step_description: str,
        examples: list[ScaleExample],
    ) -> None:
        self.category = category
        self.minimum_value = minimum_value
        self.minimum_description = minimum_description
        self.maximum_value = maximum_value
        self.maximum_description = maximum_description
        self.step_description = step_description
        self.examples = examples

        if self.minimum_value >= self.maximum_value:
            raise ValueError("minimum_value must be less than maximum_value")

        if not examples:
            raise ValueError("examples must be provided")

        if not all(isinstance(example, ScaleExample) for example in examples):
            raise ValueError("All examples must be of type ScaleExample")

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "minimum_value": self.minimum_value,
            "minimum_description": self.minimum_description,
            "maximum_value": self.maximum_value,
            "maximum_description": self.maximum_description,
            "step_description": self.step_description,
            "examples": "\n\n".join(
                [
                    f"Example {index}:\n"
                    f"task: {example.task}\n"
                    f"response: {example.response}\n"
                    f"rationale: {example.rationale}\n"
                    f"score_value: {example.score_value}"
                    for index, example in enumerate(self.examples, start=1)
                ]
            ),
        }


class SelfAskScaleScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a customizable numeric scale.
    """

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        scale_path: Optional[Path] = None,
        scale: Optional[Scale] = None,
        memory: MemoryInterface = None,
    ) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "float_scale"

        self._memory = memory if memory else DuckDBMemory()

        if not scale_path and not scale:
            raise ValueError("Either scale_path or scale must be provided.")
        if scale_path and scale:
            raise ValueError("Only one of scale_path or scale should be provided.")
        if scale_path:
            scale_args = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
            if "examples" in scale_args:
                scale_args["examples"] = [
                    ScaleExample(
                        task=example["task"],
                        response=example["response"],
                        rationale=example["rationale"],
                        score_value=example["score_value"],
                    )
                    for example in scale_args["examples"]
                ]
            self._scale = Scale(**scale_args)
        else:
            self._scale = scale

        scoring_instructions_template = PromptTemplate.from_yaml_file(SCALES_PATH / "scale_system_prompt.yaml")
        system_prompt_kwargs = self._scale.to_dict()
        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(**system_prompt_kwargs)

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The request_response scored.
                         The score_value is a value from [0,1] that is scaled based on the scorer's scale.
        """
        self.validate(request_response, task=task)

        conversation_id = str(uuid.uuid4())

        self._prompt_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        scoring_prompt = f"task: {task}\nresponse: {request_response.converted_value}"

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=scoring_prompt,
                    conversation_id=conversation_id,
                    prompt_target_identifier=self._prompt_target.get_identifier(),
                )
            ]
        )

        score = await self._send_chat_target_async(request, request_response.id)
        score.task = task
        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    @pyrit_json_retry
    async def _send_chat_target_async(self, request, request_response_id):
        response = await self._prompt_target.send_prompt_async(prompt_request=request)

        try:
            response_json = response.request_pieces[0].converted_value
            parsed_response = json.loads(response_json)
            score_value = self.scale_value_float(
                float(parsed_response["score_value"]), self._scale.minimum_value, self._scale.maximum_value
            )
            score = Score(
                score_value=str(score_value),
                score_value_description=parsed_response["description"],
                score_type=self.scorer_type,
                score_category=self._scale.category,
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

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if request_response.original_value_data_type != "text":
            raise ValueError("The original value data type must be text.")
        if not task:
            raise ValueError("Task must be provided.")
