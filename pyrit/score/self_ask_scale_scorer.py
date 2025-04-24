# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
from pathlib import Path
from typing import Optional

import yaml

from pyrit.common.path import SCALES_PATH
from pyrit.models import PromptRequestPiece, Score, SeedPrompt, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer


class SelfAskScaleScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a customizable numeric scale.
    """

    class ScalePaths(enum.Enum):
        TREE_OF_ATTACKS_SCALE = Path(SCALES_PATH, "tree_of_attacks_scale.yaml").resolve()
        TASK_ACHIEVED_SCALE = Path(SCALES_PATH, "task_achieved_scale.yaml").resolve()
        CRITERIA_SCALE = Path(SCALES_PATH, "criteria_example_scale.yaml").resolve()

    class SystemPaths(enum.Enum):
        GENERAL_SYSTEM_PROMPT = Path(SCALES_PATH, "general_system_prompt.yaml").resolve()
        RED_TEAMER_SYSTEM_PROMPT = Path(SCALES_PATH, "red_teamer_system_prompt.yaml").resolve()
        CRITERIA_SYSTEM_PROMPT = Path(SCALES_PATH, "criteria_system_prompt.yaml").resolve()

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        scale_arguments_path: Optional[Path],
        system_prompt_path: Optional[Path],
    ) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "float_scale"

        if not system_prompt_path:
            system_prompt_path = self.SystemPaths.GENERAL_SYSTEM_PROMPT.value

        if not scale_arguments_path:
            scale_arguments_path = self.ScalePaths.TREE_OF_ATTACKS_SCALE.value

        scale_args = yaml.safe_load(scale_arguments_path.read_text(encoding="utf-8"))

        self._validate_scale_arguments_set(scale_args)

        self._minimum_value = scale_args["minimum_value"]
        self._maximum_value = scale_args["maximum_value"]
        self._category = scale_args["category"]

        scoring_instructions_template = SeedPrompt.from_yaml_file(system_prompt_path)

        self._system_prompt = scoring_instructions_template.render_template_value(**scale_args)

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

        scoring_prompt = f"task: {task}\nresponse: {request_response.converted_value}"

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=scoring_prompt,
            prompt_request_data_type=request_response.converted_value_data_type,
            scored_prompt_id=request_response.id,
            category=self._category,
            task=task,
        )

        score = unvalidated_score.to_score(
            score_value=str(
                self.scale_value_float(
                    float(unvalidated_score.raw_score_value), self._minimum_value, self._maximum_value
                )
            )
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if request_response.original_value_data_type != "text":
            raise ValueError("The original value data type must be text.")
        if not task:
            raise ValueError("Task must be provided.")

    def _validate_scale_arguments_set(self, scale_args: dict):

        try:
            minimum_value = scale_args["minimum_value"]
            maximum_value = scale_args["maximum_value"]
            category = scale_args["category"]
        except KeyError as e:
            raise ValueError(f"Missing key in scale_args: {e.args[0]}") from None

        if not isinstance(minimum_value, int):
            raise ValueError(f"Minimum value must be an integer, got {type(minimum_value).__name__}.")
        if not isinstance(maximum_value, int):
            raise ValueError(f"Maximum value must be an integer, got {type(maximum_value).__name__}.")
        if minimum_value > maximum_value:
            raise ValueError("Minimum value must be less than or equal to the maximum value.")
        if not category:
            raise ValueError("Category must be set and cannot be empty.")
