# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal, Optional

from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    TrueFalseAggregatorFunc,
    TrueFalseScoreAggregator,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


RequestTextType = Literal["original_value", "converted_value", "metadata"]

class DecodingScorer(TrueFalseScorer):
    """
    Scorer that checks if the request text is in the output converted_value
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(
        supported_data_types=["text"],
        required_role=["assistant"]
    )

    def __init__(
        self,
        *,
        case_insensitive: bool = True,
        request_text_type: RequestTextType = "original_value",
        categories: Optional[list[str]] = None,
        aggregator: TrueFalseAggregatorFunc = TrueFalseScoreAggregator.OR,
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        """Initialize the DecodingScorer.

        Args:
            case_insensitive (bool): Whether to perform case-insensitive matching.
            categories (Optional[list[str]]): Optional list of categories for the score. Defaults to None.
            aggregator (TrueFalseAggregatorFunc): The aggregator function to use.
                Defaults to TrueFalseScoreAggregator.OR.
            validator (Optional[ScorerPromptValidator]): Custom validator. Defaults to None.
        """
        super().__init__(score_aggregator=aggregator, validator=validator or self._default_validator)
        self._case_insensitive = case_insensitive
        self._score_categories = categories if categories else []
        self._request_text_type = request_text_type

    async def _score_piece_async(
        self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        """Score the given request piece based on presence of the substring.

        Args:
            request_piece (PromptRequestPiece): The prompt request piece to score.
            objective (Optional[str]): The objective to evaluate against. Defaults to None.
                Currently not used for this scorer.

        Returns:
            list[Score]: A list containing a single Score object with a boolean value indicating
                whether the substring is present in the text.
        """

        memory = CentralMemory.get_memory_instance()
        user_request = memory.get_request_from_response(response=request_piece.to_prompt_request_response())

        expected_output_substring_present = False

        # The user original_value should be found in the model response converted_value, for any of the user pieces
        for user_piece in user_request.request_pieces:

            user_request_text = ""
            if self._request_text_type == "original_value":
                user_request_text = user_piece.original_value
            elif self._request_text_type == "converted_value":
                user_request_text = user_piece.converted_value
            elif self._request_text_type == "metadata":
                user_request_text = str(user_piece.prompt_metadata.get("decoded_text", ""))
            else:
                raise ValueError(f"Invalid request_text_type: {self._request_text_type}")

            if self._case_insensitive:
                expected_output_substring_present = user_request_text.lower() in request_piece.converted_value.lower()
            else:
                expected_output_substring_present = user_request_text in request_piece.converted_value

            if expected_output_substring_present:
                break   

        score = [
            Score(
                score_value=str(expected_output_substring_present),
                score_value_description="",
                score_metadata=None,
                score_type="true_false",
                score_category=self._score_categories,
                score_rationale="",
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_piece.id,
                objective=objective,
            )
        ]

        return score
