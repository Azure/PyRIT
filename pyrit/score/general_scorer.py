# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal, Optional

from pyrit.models import PromptRequestPiece
from pyrit.models.score import Score, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer


class SelfAskGeneralScorer(Scorer):
    """
    A general scorer that uses a chat target to score a prompt request piece.
    It can be configured to use different scoring types (e.g., true/false, float scale)
    It can also format the prompt using a system-level prompt and a format string.

    Params:
        chat_target (PromptChatTarget): The chat target to use for scoring.
        system_prompt (str): The system-level prompt that guides the behavior of the target LLM.
            Defaults to None.
            This can be a string or a format string template with placeholders for task and request_response.
            The system prompt or prompt or request_response needs to specify a JSON output format for the response
        prompt_format_string (str): A format string template for the prompt. Defaults to None.
            This is a string that can be formatted with task, request_response, and prompt
        scorer_type (str): The type of scorer (e.g., "true_false", "float_scale").
            Defaults to float_scale.
        score_value_output_key (str): The key in the JSON response that contains the score value.
            Defaults to "score_value".
        rationale_output_key (str): The key in the JSON response that contains the rationale.
            Defaults to "rationale".
        description_output_key (str): The key in the JSON response that contains the description.
            Defaults to "description".
        metadata_output_key (str): The key in the JSON response that contains the metadata.
            Defaults to "metadata".
        category_output_key (str): The key in the JSON response that contains the category.
            Defaults to "category".
        category (list): A list of categories for the score. Defaults to None.
        labels (list): A list of labels for the score. Defaults to None.
        min_value (int): The minimum value for float scale scoring. Defaults to 0.
        max_value (int): The maximum value for float scale scoring. Defaults to 100.
    """

    def __init__(
        self,
        chat_target: PromptChatTarget,
        system_prompt_format_string: str = None,
        prompt_format_string: str = None,
        scorer_type: Literal["true_false", "float_scale"] = "float_scale",
        score_value_output_key: str = "score_value",
        rationale_output_key: str = "rationale",
        description_output_key: str = "description",
        metadata_output_key: str = "metadata",
        category_output_key: str = "category",
        category: list = None,
        labels: list = None,
        min_value: int = 0,
        max_value: int = 100,
    ) -> None:

        self._prompt_target = chat_target
        self._system_prompt = system_prompt_format_string
        self._prompt_format_string = prompt_format_string

        if scorer_type != "true_false" and scorer_type != "float_scale":
            raise ValueError(
                f"Scorer type {scorer_type} is not a valid scorer type. Options are true_false or float_scale."
            )

        self.scorer_type = scorer_type

        self._score_category = category
        self.labels = labels
        self._min_value = min_value
        self._max_value = max_value
        self._score_value_output_key = score_value_output_key
        self._rationale_output_key = rationale_output_key
        self._description_output_key = description_output_key
        self._metadata_output_key = metadata_output_key
        self._category_output_key = category_output_key

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self.validate(request_response, task=task)
        prompt = request_response.converted_value

        system_prompt = self._system_prompt.format(task=task, prompt=prompt, request_response=request_response)

        if self._prompt_format_string:
            prompt = self._prompt_format_string.format(task=task, prompt=prompt, request_response=request_response)

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=system_prompt,
            prompt_request_value=prompt,
            prompt_request_data_type=request_response.converted_value_data_type,
            scored_prompt_id=request_response.id,
            category=self._score_category,
            task=task,
            orchestrator_identifier=request_response.orchestrator_identifier,
            score_value_output_key=self._score_value_output_key,
            rationale_output_key=self._rationale_output_key,
            description_output_key=self._description_output_key,
            metadata_output_key=self._metadata_output_key,
            category_output_key=self._category_output_key,
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
