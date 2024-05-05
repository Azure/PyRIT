# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
import yaml

from dataclasses import dataclass
import enum
from pathlib import Path
from typing import Dict, Union

from pyrit.score import Score, Scorer
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import CONTENT_CLASSIFIERS_PATH, LIKERT_SCALES_PATH, SCORING_INSTRUCTIONS_PATH
from pyrit.score.scorer import TrueFalseScorer



class ContentClassifiers(enum.Enum):
    HARMFUL_CONTENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "harmful_content.yaml").resolve()
    SENTIMENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "sentiment.yaml").resolve()


class SelfAskCategoryScorer(TrueFalseScorer):
    """A class that represents a self-ask score for text classification and scoring.

    This class is responsible for scoring text using a self-ask approach. It takes a chat target,
    a prompt template path, and classifier categories path as input.

    Args:
        prompt_template_path (ScoringInstructions): The path to the prompt template file.
        content_classifier (Union[ContentClassifiers, LikertScales]): The path to the classifier file.
        chat_target (PromptChatTarget): The chat target to interact with.

    """

    def __init__(
        self,
        chat_target: PromptChatTarget,
        content_classifier: ContentClassifiers,
    ) -> None:
        self._score_type = "true_false"

        category_file_contents = yaml.safe_load(Path(content_classifier.value).read_text(encoding="utf-8"))

        self._false_category = category_file_contents["false_category"]
        categories_as_string = self._content_classifier_to_string(category_file_contents["categories"])

        scoring_instructions_template = PromptTemplate.from_yaml_file(
            SCORING_INSTRUCTIONS_PATH / "classification_system_prompt.yaml"
        )

        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(categories=categories_as_string)

        self._chat_target: PromptChatTarget = chat_target
        self._conversation_id = str(uuid.uuid4())

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
        )

    def _content_classifier_to_string(self, categories: list[Dict[str, str]]) -> str:
        if not categories:
            raise ValueError("Impropoerly formated content classifier yaml file. No categories provided")

        category_descriptions = ""

        for category in categories:
            name = category["name"]
            desc = category["description"]

            category_descriptions += f"'{name}': {desc}\n"

        return category_descriptions

    async def score(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Scores the given text using the chat target.

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

            score_value = parsed_response["category_name"] != self._false_category

            score = Score(
                score_value=str(score_value),
                score_value_description=parsed_response["category_description"],
                scorer_type=self._score_type,
                score_category=parsed_response["category_name"],
                score_rationale=parsed_response["rationale"],
                scorer_class_identifier=self.get_identifier(),
                metadata=None,
                prompt_request_response_id=request_response.id,
            )
            return [score]

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from chat target: {response_json}") from e

    def validate(self, request_response: PromptRequestPiece):
        pass