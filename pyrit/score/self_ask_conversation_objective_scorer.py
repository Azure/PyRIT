# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pathlib
import uuid
import yaml

import enum
from pathlib import Path

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions.exception_classes import InvalidJsonException, pyrit_json_retry
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.score import Score, Scorer
from pyrit.prompt_target import PromptChatTarget


OBJECTIVE_QUESTIONS_PATH = pathlib.Path(DATASETS_PATH, "score", "conversation_objective").resolve()


class ObjectiveQuestionPaths(enum.Enum):
    EVALUATION = Path(OBJECTIVE_QUESTIONS_PATH, "eval.yaml").resolve()
    REFUSAL = Path(OBJECTIVE_QUESTIONS_PATH, "refusal.yaml").resolve()


class SelfAskObjectiveScorer(Scorer):
    """A class that represents a self-ask true/false for scoring a response against a given objective."""

    def __init__(
        self, *, chat_target: PromptChatTarget, objective_question_path: Path, memory: MemoryInterface = None
    ) -> None:
        self.scorer_type = "true_false"

        self._memory = memory if memory else DuckDBMemory()

        objective_question_contents = yaml.safe_load(objective_question_path.read_text(encoding="utf-8"))

        self._category = objective_question_contents["category"]
        objective_scorer_objective = objective_question_contents["objective_scorer_objective"]
        true_category = objective_question_contents["true_description"]
        false_category = objective_question_contents["false_description"]

        metadata = objective_question_contents["metadata"] if "metadata" in objective_question_contents else ""

        scoring_instructions_template = PromptTemplate.from_yaml_file(
            OBJECTIVE_QUESTIONS_PATH / "conversation_objective_prompt.yaml"
        )

        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
            objective_scorer_objective=objective_scorer_objective,
            true_description=true_category,
            false_description=false_category,
            metadata=metadata,
        )

        self._chat_target: PromptChatTarget = chat_target

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the TrueFalseQuestionPath.
                         The score_value is True or False based on which fits best.
                         metadata can be configured to provide additional information.
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
            score = Score(
                score_value=str(parsed_response["value"]),
                score_value_description=parsed_response["description"],
                score_type=self.scorer_type,
                score_category=self._category,
                score_rationale=parsed_response["rationale"],
                scorer_class_identifier=self.get_identifier(),
                score_metadata=parsed_response["metadata"],
                prompt_request_response_id=request_response_id,
            )
        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON response: {response_json}")

        except KeyError:
            raise InvalidJsonException(message=f"Invalid JSON response, missing Key: {response_json}")

        return score

    def validate(self, request_response: PromptRequestPiece):
        pass
