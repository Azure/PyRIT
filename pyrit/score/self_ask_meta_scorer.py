# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pathlib
import uuid
import yaml

import enum
from pathlib import Path

from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.score import Score, Scorer
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import DATASETS_PATH

META_SCORER_QUESTIONS_PATH = pathlib.Path(DATASETS_PATH, "score", "meta").resolve()


class MetaScorerQuestionPaths(enum.Enum):
    META_JUDGE_PROMPT = Path(META_SCORER_QUESTIONS_PATH, "meta_judge.yaml").resolve()


class SelfAskMetaScorer(Scorer):
    """A class that represents a self-ask meta scorer for scoring."""

    def __init__(
        self, *, chat_target: PromptChatTarget, meta_scorer_question_path: Path, memory: MemoryInterface = None
    ) -> None:
        self.scorer_type = "true_false"

        self._memory = memory if memory else DuckDBMemory()

        meta_scorer_question_contents = yaml.safe_load(meta_scorer_question_path.read_text(encoding="utf-8"))

        self._category = meta_scorer_question_contents["category"]
        true_category = meta_scorer_question_contents["true_description"]
        false_category = meta_scorer_question_contents["false_description"]

        metadata = meta_scorer_question_contents["metadata"] if "metadata" in meta_scorer_question_contents else ""

        scoring_instructions_template = PromptTemplate.from_yaml_file(
            META_SCORER_QUESTIONS_PATH / "meta_scorer_prompt.yaml"
        )

        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
            true_description=true_category, false_description=false_category, metadata=metadata
        )

        self._chat_target: PromptChatTarget = chat_target
        self._conversation_id = str(uuid.uuid4())

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
        )

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the MetaScorerQuestionPath.
                         The score_value is True or False based on which fits best.
                         metadata can be configured to provide additional information.
        """

        self.validate(request_response)

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=request_response.converted_value,
                    conversation_id=self._conversation_id,
                    prompt_target_identifier=self._chat_target.get_identifier(),
                )
            ]
        )

        response = await self._chat_target.send_prompt_async(prompt_request=request)
        response_json = response.request_pieces[0].converted_value

        try:
            parsed_response = json.loads(response_json)

            score = Score(
                score_value=str(parsed_response["value"]),
                score_value_description=parsed_response["description"],
                score_type=self.scorer_type,
                score_category=self._category,
                score_rationale=parsed_response["rationale"],
                scorer_class_identifier=self.get_identifier(),
                score_metadata=None,
                prompt_request_response_id=request_response.id,
            )

            self._memory.add_scores_to_memory(scores=[score])
            return [score]

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from chat target: {response_json}") from e

    def validate(self, request_response: PromptRequestPiece):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Self-ask meta scorer only supports text data type")
