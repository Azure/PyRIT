# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
from pathlib import Path
from typing import Optional, Literal

import yaml
from langsmith import expect

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestPiece, SeedPrompt
from pyrit.models.score import Score, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer

AH_SCORERS_PATH = Path(DATASETS_PATH, "score", "ah_scorers").resolve()


class EvaluatorQuestionPaths(enum.Enum):
    EVALUATOR = Path(AH_SCORERS_PATH, "evaluator_system_prompt.yaml").resolve()

class EvaluatorQuestion:
    """
    A class that represents a true/false question.

    This is sent to an LLM and can be used as an alternative to a yaml file from TrueFalseQuestionPaths.
    """

    def __init__(
            self, *, evaluation_criteria: str = "", category: str = "", metadata: Optional[str] = ""
    ):
        self.evaluation_criteria = evaluation_criteria
        self.category = category
        self.metadata = metadata

        self._keys = ["category", "evaluation_criteria"]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        # Define which keys should be included when iterating
        return iter(self._keys)


class Evaluator(Scorer):
    """A class that represents a self-ask true/false for scoring."""

    def __init__(
            self,
            *,
            chat_target: PromptChatTarget,
            evaluator_yaml_path: Optional[Path] = None,
            evaluator_question: Optional[EvaluatorQuestion] = None,
            evaluator_system_prompt_path: Optional[Path] = None,
            additional_evaluator_variables: Optional[dict] = None,
            scorer_type: Literal["true_false", "float_scale"] = "float_scale",
    ) -> None:
        self._prompt_target = chat_target
        self.scorer_type = scorer_type
        self._additional_evaluator_variables = additional_evaluator_variables or {}

        if not evaluator_yaml_path and not evaluator_question:
            raise ValueError("Either true_false_question_path or true_false_question must be provided.")
        if evaluator_yaml_path and evaluator_question:
            raise ValueError("Only one of true_false_question_path or true_false_question should be provided.")
        if evaluator_yaml_path:
            evaluator_question = yaml.safe_load(evaluator_yaml_path.read_text(encoding="utf-8"))

        for key in ["category", "evaluation_criteria"]:
            if key not in evaluator_question:
                raise ValueError(f"{key} must be provided in true_false_question.")

        self._score_category = evaluator_question["category"]
        evaluation_criteria = evaluator_question["evaluation_criteria"]

        metadata = evaluator_question["metadata"] if "metadata" in evaluator_question else ""

        evaluator_system_prompt_path = (
            evaluator_system_prompt_path
            if evaluator_system_prompt_path
            else AH_SCORERS_PATH / "evaluator_system_prompt.yaml"
        )

        scoring_instructions_template = SeedPrompt.from_yaml_file(evaluator_system_prompt_path)

        self._system_prompt = scoring_instructions_template.render_template_value(
            evaluation_criteria=evaluation_criteria, metadata=metadata
        )

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
                Currently not supported for this scorer.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the TrueFalseQuestionPath.
                         The score_value is True or False based on which fits best.
                         metadata can be configured to provide additional information.
        """

        self.validate(request_response, task=task)

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=request_response.converted_value,
            prompt_request_data_type=request_response.converted_value_data_type,
            scored_prompt_id=request_response.id,
            category=self._score_category,
            task=task,
            request_prompt=request_response.original_value,
            expected_output=request_response.expected_output,
            additional_evaluator_variables=self._additional_evaluator_variables,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value, expected_output=request_response.expected_output)

        self._memory.add_scores_to_memory(scores=[score])
        return [score]


    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass
