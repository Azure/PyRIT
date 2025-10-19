# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from pyrit.common.path import LIKERT_SCALES_PATH
from pyrit.models import MessagePiece, Score, SeedPrompt, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator

logger = logging.getLogger(__name__)


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
    DECEPTION_SCALE = Path(LIKERT_SCALES_PATH, "deception.yaml").resolve()


class SelfAskLikertScorer(FloatScaleScorer):
    """
    A class that represents a "self-ask" score for text scoring for a likert scale.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["text"])

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        likert_scale_path: Union[str, Path],
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        super().__init__(validator=validator or self._default_validator)

        likert_scale_path = self._verify_and_resolve_path(likert_scale_path)

        self._prompt_target = chat_target

        self.set_likert_scale_system_prompt(likert_scale_path=likert_scale_path)

    def set_likert_scale_system_prompt(self, likert_scale_path: Path):
        """
        Sets the Likert scale to use for scoring.

        Args:
            likert_scale_path (Path): The path to the YAML file containing the Likert scale description.
        """
        likert_scale = yaml.safe_load(likert_scale_path.read_text(encoding="utf-8"))

        if likert_scale["category"]:
            self._score_category = likert_scale["category"]
        else:
            raise ValueError(f"Improperly formatted likert scale yaml file. Missing category in {likert_scale_path}.")

        likert_scale = self._likert_scale_description_to_string(likert_scale["scale_descriptions"])

        self._scoring_instructions_template = SeedPrompt.from_yaml_file(
            LIKERT_SCALES_PATH / "likert_system_prompt.yaml"
        )

        self._system_prompt = self._scoring_instructions_template.render_template_value(
            likert_scale=likert_scale, category=self._score_category
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
            raise ValueError("Improperly formatted Likert scale yaml file. No likert scale_descriptions provided")

        likert_scale_description = ""

        for description in descriptions:
            name = description["score_value"]
            desc = description["description"]

            if int(name) < 0 or int(name) > 5:
                raise ValueError(
                    "Improperly formatted Likert scale yaml file. Likert scale values must be between 1 and 5"
                )

            likert_scale_description += f"'{name}': {desc}\n"

        return likert_scale_description

    async def _score_piece_async(
        self, message_piece: MessagePiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Scores the given message_piece using "self-ask" for the chat target.

        Args:
            message_piece (MessagePiece): The prompt request piece containing the text to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
                Currently not supported for this scorer.

        Returns:
            list[Score]: The message_piece scored.
                         The category is configured from the likert_scale.
                         The score_value is a value from [0,1] that is scaled from the likert scale.
        """
        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=message_piece.converted_value,
            prompt_request_data_type=message_piece.converted_value_data_type,
            scored_prompt_id=message_piece.id,
            category=self._score_category,
            objective=objective,
        )

        score = unvalidated_score.to_score(
            score_value=str(self.scale_value_float(float(unvalidated_score.raw_score_value), 1, 5)),
            score_type="float_scale",
        )

        score.score_metadata = {"likert_value": int(unvalidated_score.raw_score_value)}

        return [score]
