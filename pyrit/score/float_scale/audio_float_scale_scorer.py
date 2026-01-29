# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.identifiers import ScorerIdentifier
from pyrit.models import MessagePiece, Score
from pyrit.score.audio_scorer import _BaseAudioScorer
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


class AudioFloatScaleScorer(FloatScaleScorer, _BaseAudioScorer):
    """
    A scorer that processes audio files by transcribing them and scoring the transcript.

    The AudioFloatScaleScorer transcribes audio to text using Azure Speech-to-Text,
    then scores the transcript using a FloatScaleScorer.

    Example usage:
        ```python
        text_scorer = SelfAskGeneralFloatScaleScorer(
            chat_target=OpenAIChatTarget(),
            system_prompt_format_string=harmful_content_rubric,
            min_value=1,
            max_value=5,
        )

        audio_scorer = AudioFloatScaleScorer(text_capable_scorer=text_scorer)
        score = await audio_scorer.score_async(audio_message)
        ```
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["audio_path"])

    def __init__(
        self,
        *,
        text_capable_scorer: FloatScaleScorer,
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        """
        Initialize the AudioFloatScaleScorer.

        Args:
            text_capable_scorer: A FloatScaleScorer capable of processing text.
                This scorer will be used to evaluate the transcribed audio content.
            validator: Validator for the scorer. Defaults to audio_path data type validator.
        """
        _BaseAudioScorer.__init__(self, text_capable_scorer=text_capable_scorer)
        FloatScaleScorer.__init__(self, validator=validator or self._default_validator)

    def _build_identifier(self) -> ScorerIdentifier:
        """
        Build the scorer evaluation identifier for this scorer.

        Returns:
            ScorerIdentifier: The identifier for this scorer.
        """
        return self._create_identifier(
            sub_scorers=[self.text_scorer],
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Score an audio file by transcribing it and scoring the transcript.

        Args:
            message_piece: The message piece containing the audio file path.
            objective: Optional objective description for scoring.

        Returns:
            List of scores from evaluating the transcribed audio.
        """
        scores = await self._score_audio_async(message_piece=message_piece, objective=objective)

        if not scores:
            # No transcript or empty transcript - return a 0.0 score
            piece_id = message_piece.id if message_piece.id is not None else message_piece.original_prompt_id
            return [
                Score(
                    score_value="0.0",
                    score_value_description="No audio content to score (empty or no transcript)",
                    score_type="float_scale",
                    score_category=["audio"],
                    score_rationale="Audio file had no transcribable content",
                    scorer_class_identifier=self.get_identifier(),
                    message_piece_id=piece_id,
                )
            ]

        # Update rationale to indicate this was from audio transcription
        for score in scores:
            score.score_rationale = f"Audio transcript scored: {score.score_rationale}"

        return scores
