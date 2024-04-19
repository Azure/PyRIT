# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pyrit.interfaces import SupportTextClassification
from pyrit.models import Score
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget


@dataclass
class Category:
    name: str
    description: str


# System prompt templates
CLASSIFICATION_SYSTEM_PROMPT = Path(__file__).parent / "scoring_instructions" / "classification_system_prompt.yaml"
LIKERT_SYSTEM_PROMPT = Path(__file__).parent / "scoring_instructions" / "likert_system_prompt.yaml"

# Classifier / Likert scale categories
BIOCHEM_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "biochem.yaml"
CURRENT_EVENTS_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "current_events.yaml"
CYBER_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "cyber.yaml"
FAIRNESS_BIAS_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "fairness_bias.yaml"
HATE_SPEECH_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "hate_speech.yaml"
PERSUASION_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "persuasion.yaml"
PHISH_EMAILS_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "phish_emails.yaml"
POLITICAL_MISINFO_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "political_misinfo.yaml"
PROMPT_INJECTION_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "prompt_injection_detector.yaml"
QUESTION_ANSWERING_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "question_answering.yaml"
SENTIMENT_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "sentiment.yaml"
SEXUAL_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "sexual.yaml"
VIOLENCE_CLASSIFIER = Path(__file__).parent / "content_classifiers" / "violence.yaml"


class SelfAskScore(SupportTextClassification):
    """A class that represents a self-ask score for text classification and scoring.

    This class is responsible for scoring text using a self-ask approach. It takes a chat target,
    a prompt template path, and classifier categories path as input.

    Args:
        prompt_template_path (pathlib.Path | str): The path to the prompt template file.
        content_classifier (pathlib.Path | str): The path to the classifier categories file.
        chat_target (PromptChatTarget): The chat target to interact with.

    Attributes:
        _chat_target (PromptChatTarget): The chat target used for scoring.
        _system_prompt (Prompt): The system prompt generated from the prompt template and categories.

    """

    def __init__(
        self,
        prompt_template_path: Union[str, Path],
        content_classifier: Union[str, Path],
        chat_target: PromptChatTarget,
    ) -> None:
        # Create the system prompt with the categories
        categories_as_string = ""
        category_file_contents = yaml.safe_load(Path(content_classifier).read_text(encoding="utf-8"))
        for k, v in category_file_contents.items():
            category = Category(name=k, description=v)
            categories_as_string += f"'{category.name}': {category.description}\n"
        prompt_template = PromptTemplate.from_yaml_file(Path(prompt_template_path))
        self._system_prompt = prompt_template.apply_custom_metaprompt_parameters(categories=categories_as_string)

        self._chat_target: PromptChatTarget = chat_target
        self._conversation_id = str(uuid.uuid4())
        self._normalizer_id = None  # Normalizer not used
        self.labels = {"scorer": "self_ask_scorer"}

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
            labels=self.labels,
        )

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
                    original_prompt_text=text,
                    conversation_id=self._conversation_id,
                    labels=self.labels,
                    prompt_target_identifier=self._chat_target.get_identifier(),
                )
            ]
        )

        response_text = self._chat_target.send_prompt(prompt_request=request).request_pieces[0].converted_prompt_text

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


class SelfAskGptClassifier(SelfAskScore):
    def __init__(
        self,
        content_classifier: Union[str, Path],
        chat_target: PromptChatTarget,
    ) -> None:

        super().__init__(
            prompt_template_path=CLASSIFICATION_SYSTEM_PROMPT,
            content_classifier=content_classifier,
            chat_target=chat_target,
        )


class SelfAskGptLikertScale(SelfAskScore):
    def __init__(
        self,
        content_classifier: Union[str, Path],
        chat_target: PromptChatTarget,
    ) -> None:

        super().__init__(
            prompt_template_path=LIKERT_SYSTEM_PROMPT,
            content_classifier=content_classifier,
            chat_target=chat_target,
        )
