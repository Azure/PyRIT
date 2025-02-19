# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
from pathlib import Path
from typing import Optional

import yaml

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestPiece, SeedPrompt
from pyrit.models.score import Score, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer

TRUE_FALSE_QUESTIONS_PATH = Path(DATASETS_PATH, "score", "true_false_question").resolve()


class TrueFalseQuestionPaths(enum.Enum):
    CURRENT_EVENTS = Path(TRUE_FALSE_QUESTIONS_PATH, "current_events.yaml").resolve()
    GROUNDED = Path(TRUE_FALSE_QUESTIONS_PATH, "grounded.yaml").resolve()
    PROMPT_INJECTION = Path(TRUE_FALSE_QUESTIONS_PATH, "prompt_injection.yaml").resolve()
    QUESTION_ANSWERING = Path(TRUE_FALSE_QUESTIONS_PATH, "question_answering.yaml").resolve()
    GANDALF = Path(TRUE_FALSE_QUESTIONS_PATH, "gandalf.yaml").resolve()


class TrueFalseQuestion:
    """
    A class that represents a true/false question.

    This is sent to an LLM and can be used as an alternative to a yaml file from TrueFalseQuestionPaths.
    """

    def __init__(
        self, *, true_description: str, false_description: str = "", category: str = "", metadata: Optional[str] = ""
    ):
        self.true_description = true_description

        self.false_description = (
            false_description or "Anything that does not fulfill or is not related to the true description."
        )

        self.category = category
        self.metadata = metadata

        self._keys = ["category", "true_description", "false_description"]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        # Define which keys should be included when iterating
        return iter(self._keys)


class SelfAskTrueFalseScorer(Scorer):
    """A class that represents a self-ask true/false for scoring."""

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        true_false_question_path: Optional[Path] = None,
        true_false_question: Optional[TrueFalseQuestion] = None,
        true_false_system_prompt_path: Optional[Path] = None,
    ) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "true_false"

        if not true_false_question_path and not true_false_question:
            raise ValueError("Either true_false_question_path or true_false_question must be provided.")
        if true_false_question_path and true_false_question:
            raise ValueError("Only one of true_false_question_path or true_false_question should be provided.")
        if true_false_question_path:
            true_false_question = yaml.safe_load(true_false_question_path.read_text(encoding="utf-8"))

        for key in ["category", "true_description", "false_description"]:
            if key not in true_false_question:
                raise ValueError(f"{key} must be provided in true_false_question.")

        self._score_category = true_false_question["category"]
        true_category = true_false_question["true_description"]
        false_category = true_false_question["false_description"]

        metadata = true_false_question["metadata"] if "metadata" in true_false_question else ""

        true_false_system_prompt_path = (
            true_false_system_prompt_path
            if true_false_system_prompt_path
            else TRUE_FALSE_QUESTIONS_PATH / "true_false_system_prompt.yaml"
        )

        scoring_instructions_template = SeedPrompt.from_yaml_file(true_false_system_prompt_path)

        self._system_prompt = scoring_instructions_template.render_template_value(
            true_description=true_category, false_description=false_category, metadata=metadata
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
            orchestrator_identifier=request_response.orchestrator_identifier,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass
