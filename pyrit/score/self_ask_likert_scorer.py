# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
import yaml
import enum

from pathlib import Path
from typing import Dict


from pyrit.score import Score, Scorer
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import LIKERT_SCALES_PATH


class LikertScalePaths(enum.Enum):
    CYBER_SCALE = Path(LIKERT_SCALES_PATH, "cyber.yaml").resolve()
    FAIRNESS_BIAS_SCALE = Path(LIKERT_SCALES_PATH, "fairness_bias.yaml").resolve()
    HARM_SCALE = Path(LIKERT_SCALES_PATH, "harm.yaml").resolve()
    HATE_SPEECH_SCALE = Path(LIKERT_SCALES_PATH, "hate_speech.yaml").resolve()
    MISINFORMATION_SCALE = Path(LIKERT_SCALES_PATH, "misinformation.yaml").resolve()
    PERSUASION_SCALE = Path(LIKERT_SCALES_PATH, "persuasion.yaml").resolve()
    PHISHING_SCALE = Path(LIKERT_SCALES_PATH, "phishing.yaml").resolve()
    SEXUAL_SCALE = Path(LIKERT_SCALES_PATH, "sexual.yaml").resolve()
    VIOLENCE_SCALE = Path(LIKERT_SCALES_PATH, "violence.yaml").resolve()


class SelfAskLikertScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a likert scale.
    """

    def __init__(
        self,
        chat_target: PromptChatTarget,
        likert_scale_path: Path,
    ) -> None:

        self.scorer_type = "float_scale"

        likert_scale = yaml.safe_load(likert_scale_path.read_text(encoding="utf-8"))

        if likert_scale["category"]:
            self._score_category = likert_scale["category"]
        else:
            raise ValueError(f"Impropoerly formated likert scale yaml file. Missing category in {likert_scale_path}.")

        likert_scale = self._likert_scale_description_to_string(likert_scale["scale_descriptions"])

        scoring_instructions_template = PromptTemplate.from_yaml_file(LIKERT_SCALES_PATH / "likert_system_prompt.yaml")
        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
            likert_scale=likert_scale, category=self._score_category
        )

        self._chat_target: PromptChatTarget = chat_target
        self._conversation_id = str(uuid.uuid4())

        self._chat_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
        )

    def _likert_scale_description_to_string(self, descriptions: list[Dict[str, str]]) -> str:
        """
        Converts the Likert scales to a string representation to be put in a system prompt.

        Args:
            descriptions: list[Dict[str, str]]: The Likert scale to use.

        Returns:
            str: The string representation of the Likert scale.
        """
        if not descriptions:
            raise ValueError("Impropoerly formated Likert scale yaml file. No likert scale_descriptions provided")

        likert_scale_description = ""

        for description in descriptions:
            name = description["score_value"]
            desc = description["description"]

            if int(name) < 0 or int(name) > 5:
                raise ValueError(
                    "Impropoerly formated Likert scale yaml file. Likert scale values must be between 1 and 5"
                )

            likert_scale_description += f"'{name}': {desc}\n"

        return likert_scale_description

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the likert_scale.
                         The score_value is a value from [0,1] that is scaled from the likert scale.
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

            score_value = self.scale_value_float(float(parsed_response["score_value"]), 1, 5)

            score = Score(
                score_value=str(score_value),
                score_value_description=parsed_response["description"],
                score_type=self.scorer_type,
                score_category=self._score_category,
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
