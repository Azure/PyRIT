# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
from typing import Dict, Optional
import uuid
import yaml

from pathlib import Path

from pyrit.common.path import DATASETS_PATH
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Score, Scorer, UnvalidatedScore

TRUE_FALSE_QUESTIONS_PATH = Path(DATASETS_PATH, "score", "true_false_question").resolve()


class TrueFalseQuestionPaths(enum.Enum):
    CURRENT_EVENTS = Path(TRUE_FALSE_QUESTIONS_PATH, "current_events.yaml").resolve()
    GROUNDED = Path(TRUE_FALSE_QUESTIONS_PATH, "grounded.yaml").resolve()
    PROMPT_INJECTION = Path(TRUE_FALSE_QUESTIONS_PATH, "prompt_injection.yaml").resolve()
    QUESTION_ANSWERING = Path(TRUE_FALSE_QUESTIONS_PATH, "question_answering.yaml").resolve()
    GANDALF = Path(TRUE_FALSE_QUESTIONS_PATH, "gandalf.yaml").resolve()


class SelfAskTrueFalseScorer(Scorer):
    """A class that represents a self-ask true/false for scoring."""

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        true_false_question_path: Optional[Path] = None,
        true_false_question_contents: Optional[Dict[str, str]] = None,
        true_false_system_prompt_path: Optional[Path] = None,
        memory: MemoryInterface = None,
    ) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "true_false"

        self._memory = memory if memory else DuckDBMemory()

        # Ensure _prompt_target uses the same memory interface as the scorer.
        if self._prompt_target:
            self._prompt_target._memory = self._memory

        if not true_false_question_path and not true_false_question_contents:
            raise ValueError("Either true_false_question_path or true_false_question_contents must be provided.")
        if true_false_question_path and true_false_question_contents:
            raise ValueError("Only one of true_false_question_path or true_false_question_contents should be provided.")
        if true_false_question_path:
            true_false_question_contents = yaml.safe_load(true_false_question_path.read_text(encoding="utf-8"))

        for key in ["category", "true_description", "false_description"]:
            if key not in true_false_question_contents:
                raise ValueError(f"{key} must be provided in true_false_question_contents.")

        self._score_category = true_false_question_contents["category"]
        true_category = true_false_question_contents["true_description"]
        false_category = true_false_question_contents["false_description"]

        metadata = true_false_question_contents["metadata"] if "metadata" in true_false_question_contents else ""

        true_false_system_prompt_path = (
            true_false_system_prompt_path
            if true_false_system_prompt_path
            else TRUE_FALSE_QUESTIONS_PATH / "true_false_system_prompt.yaml"
        )

        scoring_instructions_template = PromptTemplate.from_yaml_file(true_false_system_prompt_path)

        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
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

        unvalidated_score: UnvalidatedScore = await self.send_chat_target_async(
            prompt_target=self._prompt_target,
            scorer_llm_request=request,
            scored_prompt_id=request_response.id,
            category=self._score_category,
            task=task,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if task:
            raise ValueError("This scorer does not support tasks")
