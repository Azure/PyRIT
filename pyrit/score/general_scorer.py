# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import PromptRequestPiece
from pyrit.models.score import Score, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer


class SelfAskGeneralScorer(Scorer):
    """
    A general scorer that uses a chat target to score a prompt request piece.
    It can be configured to use different scoring types (e.g., true/false, float scale) and formats.
    Params:
        chat_target (PromptChatTarget): The chat target to use for scoring.
        system_prompt (str): The system-level prompt that guides the behavior of the target LLM. Defaults to None.
        prompt_fstring_format (str): The format string for the prompt. Defaults to None.
        scorer_type (str): The type of scorer (e.g., "true_false", "float_scale"). defaults to float_scale.
        category (list): A list of categories for the score.
        labels (list): A list of labels for the score.
        min_value (int): The minimum value for float scale scoring.
        max_value (int): The maximum value for float scale scoring.
        output_keys (dict): A dictionary of output keys for the score response.
    """

    def __init__(
        self,
        chat_target: PromptChatTarget,
        system_prompt: Optional[str] = None,
        prompt_fstring_format: str = None,
        scorer_type: str = "float_scale",
        category: list = None,
        labels: list = None,
        min_value: Optional[int] = 0,
        max_value: Optional[int] = 100,
        output_keys: Optional[dict] = None,
    ) -> None:
        self._prompt_target = chat_target
        self._system_prompt = system_prompt
        self.prompt_fstring_format = prompt_fstring_format
        self.scorer_type = scorer_type
        self._score_category = category
        self.labels = labels
        self._min_value = min_value
        self._max_value = max_value
        self.prompt_fstring_format = prompt_fstring_format

        # Default output keys
        default_output_keys = {
            "score_value": "score_value",
            "rationale": "rationale",
            "metadata": "metadata",
            "description": "description",
        }

        # Merge default keys with provided keys (if any)
        self._output_keys = {**default_output_keys, **(output_keys or {})}

        print("SelfAskGeneralScorer initialized")

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self.validate(request_response, task=task)

        if self.prompt_fstring_format:
            full_prompt = self.prompt_fstring_format.format(task=task)

            converted_response = full_prompt
            if self._system_prompt:
                converted_response = f"{self._system_prompt} {full_prompt}"
        else:
            converted_response = self._system_prompt

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=converted_response,
            prompt_request_value=request_response.converted_value,
            prompt_request_data_type=request_response.converted_value_data_type,
            scored_prompt_id=request_response.id,
            category=self._score_category,
            task=task,
            orchestrator_identifier=request_response.orchestrator_identifier,
            output_keys=self._output_keys,
        )

        if self.scorer_type == "true_false":

            score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)

        elif self.scorer_type == "float_scale":

            score = unvalidated_score.to_score(
                score_value=str(
                    self.scale_value_float(float(unvalidated_score.raw_score_value), self._min_value, self._max_value)
                )
            )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        pass
