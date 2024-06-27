# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import json
import uuid
import yaml

from pathlib import Path

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions.exception_classes import InvalidJsonException, pyrit_json_retry
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Score, Scorer

TRUE_FALSE_QUESITIONS_PATH = Path(DATASETS_PATH, "score", "true_false_question").resolve()


class TrueFalseQuestionPaths(enum.Enum):
    CURRENT_EVENTS = Path(TRUE_FALSE_QUESITIONS_PATH, "current_events.yaml").resolve()
    GROUNDED = Path(TRUE_FALSE_QUESITIONS_PATH, "grounded.yaml").resolve()
    PROMPT_INJECTION = Path(TRUE_FALSE_QUESITIONS_PATH, "prompt_injection.yaml").resolve()
    QUESTION_ANSWERING = Path(TRUE_FALSE_QUESITIONS_PATH, "question_answering.yaml").resolve()
    GANDALF = Path(TRUE_FALSE_QUESITIONS_PATH, "gandalf.yaml").resolve()


class SelfAskTrueFalseScorer(Scorer):
    """A class that represents a self-ask true/false for scoring."""

    def __init__(
        self, *, chat_target: PromptChatTarget, true_false_question_path: Path, memory: MemoryInterface = None
    ) -> None:
        self.scorer_type = "true_false"

        self._memory = memory if memory else DuckDBMemory()

        true_false_question_contents = yaml.safe_load(true_false_question_path.read_text(encoding="utf-8"))

        self._category = true_false_question_contents["category"]
        true_category = true_false_question_contents["true_description"]
        false_category = true_false_question_contents["false_description"]

        metadata = true_false_question_contents["metadata"] if "metadata" in true_false_question_contents else ""

        scoring_instructions_template = PromptTemplate.from_yaml_file(
            TRUE_FALSE_QUESITIONS_PATH / "true_false_system_prompt.yaml"
        )

        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
            true_description=true_category, false_description=false_category, metadata=metadata
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
                    original_value_data_type=request_response.original_value_data_type,
                    converted_value=request_response.converted_value,
                    converted_value_data_type=request_response.converted_value_data_type,
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
