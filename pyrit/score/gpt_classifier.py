# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import uuid
import yaml

from dataclasses import dataclass
import enum
from pathlib import Path
from typing import Union

from pyrit.score import Score, SupportTextClassification
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import CONTENT_CLASSIFIERS_PATH, LIKERT_SCALES_PATH, SCORING_INSTRUCTIONS_PATH
from pyrit.score.support_classification import SupportImageClassification


logger = logging.getLogger(__name__)


@dataclass
class Category:
    name: str
    description: str


class ScoringInstructions(enum.Enum):
    CLASSIFICATION_SYSTEM_PROMPT = Path(SCORING_INSTRUCTIONS_PATH, "classification_system_prompt.yaml").resolve()
    LIKERT_SYSTEM_PROMPT = Path(SCORING_INSTRUCTIONS_PATH, "likert_system_prompt.yaml").resolve()


class ContentClassifiers(enum.Enum):
    BIAS_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "bias.yaml").resolve()
    CURRENT_EVENTS_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "current_events.yaml").resolve()
    GROUNDEDNESS_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "grounded.yaml").resolve()
    HARMFUL_CONTENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "harmful_content.yaml").resolve()
    PROMPT_INJECTION_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "prompt_injection_detector.yaml").resolve()
    QUESTION_ANSWERING_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "question_answering.yaml").resolve()
    REFUSAL_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "refusal.yaml").resolve()
    SENTIMENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "sentiment.yaml").resolve()
    SEXUAL_CONTENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "sexual_content.yaml").resolve()


class LikertScales(enum.Enum):
    CYBER_SCALE = Path(LIKERT_SCALES_PATH, "cyber.yaml").resolve()
    FAIRNESS_BIAS_SCALE = Path(LIKERT_SCALES_PATH, "fairness_bias.yaml").resolve()
    HARM_SCALE = Path(LIKERT_SCALES_PATH, "harm.yaml").resolve()
    HATE_SPEECH_SCALE = Path(LIKERT_SCALES_PATH, "hate_speech.yaml").resolve()
    PERSUASION_SCALE = Path(LIKERT_SCALES_PATH, "persuasion.yaml").resolve()
    PHISH_EMAILS_SCALE = Path(LIKERT_SCALES_PATH, "phish_emails.yaml").resolve()
    POLITICAL_MISINFO_SCALE = Path(LIKERT_SCALES_PATH, "political_misinfo.yaml").resolve()
    SEXUAL_SCALE = Path(LIKERT_SCALES_PATH, "sexual.yaml").resolve()
    VIOLENCE_SCALE = Path(LIKERT_SCALES_PATH, "violence.yaml").resolve()


class SelfAskScorer:
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
        prompt_template_path: ScoringInstructions,
        content_classifier: Union[ContentClassifiers, LikertScales, Path],
        chat_target: PromptChatTarget,
    ) -> None:
        # Create the system prompt with the categories
        categories_as_string = ""
        if isinstance(content_classifier, Path):
            content_classifier_path = content_classifier
        else:
            content_classifier_path = Path(content_classifier.value)
        category_file_contents = yaml.safe_load(content_classifier_path.read_text(encoding="utf-8"))
        for k, v in category_file_contents.items():
            category = Category(name=k, description=v)
            categories_as_string += f"'{category.name}': {category.description}\n"
        prompt_template = PromptTemplate.from_yaml_file(Path(prompt_template_path.value))
        self._system_prompt = prompt_template.apply_custom_metaprompt_parameters(categories=categories_as_string)

        self._chat_target: PromptChatTarget = chat_target
        self._conversation_id = str(uuid.uuid4())
        self._normalizer_id = None  # Normalizer not used

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
        )

class SelfAskTextScorer(SelfAskScorer, SupportTextClassification):
    # @tenacity.retry(wait=tenacity.wait_fixed(1), stop=tenacity.stop_after_attempt(8))
    def score_text(self, text: str) -> Score:
        """
        Scores the given text using the chat target.

        Args:
            text (str): The text to be scored.

        Returns:
            Score: An object containing the score information.

        Raises:
            ValueError: If the response from the chat target is not a valid JSON.
        """

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=text,
                    conversation_id=self._conversation_id,
                    prompt_target_identifier=self._chat_target.get_identifier(),
                )
            ]
        )

        response_text = self._chat_target.send_prompt(prompt_request=request).request_pieces[0].converted_value

        try:
            parsed_response = json.loads(response_text)

            score = Score(
                score_type="str",
                score_value=parsed_response["category_name"],
                score_description=parsed_response["category_description"],
                score_explanation=parsed_response["rationale"],
                raw_input_score_text=text,
                raw_output_score_text=response_text,
            )
            return score

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from chat target: {response_text}") from e


class SelfAskImageScorer(SelfAskScorer, SupportImageClassification):
    # @tenacity.retry(wait=tenacity.wait_fixed(1), stop=tenacity.stop_after_attempt(8))
    async def score_image(self, image_path: str) -> Score:
        """
        Scores the given text using the chat target.

        Args:
            text (str): The text to be scored.

        Returns:
            Score: An object containing the score information.

        Raises:
            ValueError: If the response from the chat target is not a valid JSON.
        """

        request_pieces = [
            PromptRequestPiece(
                role="user",
                conversation_id=self._conversation_id,
                original_value=str(image_path),
                original_value_data_type="image_path",
                converted_value_data_type="image_path",
            ),
        ]

        prompt_request_response = PromptRequestResponse(request_pieces=request_pieces)

        response = await self._chat_target.send_prompt_async(prompt_request=prompt_request_response)
        
        try:
            logger.info(f"Received scoring response: {response.request_pieces[0].converted_value}")
            parsed_response = json.loads(response.request_pieces[0].converted_value)

            score = Score(
                score_type="str",
                score_value=parsed_response["category_name"],
                score_description=parsed_response["category_description"],
                score_explanation=parsed_response["rationale"],
                raw_input_score_text=str(image_path),
                raw_output_score_text=response.request_pieces[0].converted_value,
            )
            return score

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON response from chat target: {response.request_pieces[0].converted_value}"
            ) from e


class SelfAskBinaryImageScorer(SelfAskImageScorer):
    def __init__(
        self,
        prompt_template_path: ScoringInstructions,
        content_classifier: ContentClassifiers,
        chat_target: PromptChatTarget,
        success_categories: list[str],
    ) -> None:

        super().__init__(
            prompt_template_path=prompt_template_path,
            content_classifier=content_classifier,
            chat_target=chat_target,
        )
        self._success_categories = success_categories
    
    async def score_image(self, image_path: str) -> Score:
        score = await super().score_image(image_path)
        return Score(
            score_type="bool",
            score_value=bool(score.score_value in self._success_categories),
            score_description=score.score_description,
            score_explanation=score.score_explanation,
        )


class SelfAskGptClassifier(SelfAskTextScorer):
    def __init__(
        self,
        content_classifier: ContentClassifiers,
        chat_target: PromptChatTarget,
    ) -> None:

        super().__init__(
            prompt_template_path=ScoringInstructions.CLASSIFICATION_SYSTEM_PROMPT,
            content_classifier=content_classifier,
            chat_target=chat_target,
        )


class SelfAskGptLikertScale(SelfAskTextScorer):
    def __init__(
        self,
        content_classifier: LikertScales,
        chat_target: PromptChatTarget,
    ) -> None:

        super().__init__(
            prompt_template_path=ScoringInstructions.LIKERT_SYSTEM_PROMPT,
            content_classifier=content_classifier,
            chat_target=chat_target,
        )
