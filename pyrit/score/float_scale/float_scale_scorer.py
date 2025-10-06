# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Dict, Optional
from uuid import UUID

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import Score
from pyrit.models.literals import PromptDataType
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import UnvalidatedScore
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score.scorer import Scorer


class FloatScaleScorer(Scorer):
    """
    Base class for scorers that return floating-point scores in the range [0, 1].

    This scorer evaluates prompt responses and returns numeric scores indicating the degree
    to which a response exhibits certain characteristics. Each piece in a request response
    is scored independently, returning one score per piece.
    """

    def __init__(self, *, validator) -> None:
        super().__init__(validator=validator)

    def validate_return_scores(self, scores: list[Score]):
        for score in scores:
            if not (0 <= score.get_value() <= 1):
                raise ValueError("FloatScaleScorer score value must be between 0 and 1.")

    async def _score_async(
        self, request_response: PromptRequestResponse, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Score the given request response asynchronously.

        For FloatScaleScorer, the scoring is based on a float scale from 0 to 1. It scores every piece
        of the request_response.

        Args:
            request_response (PromptRequestResponse): The prompt request response to score.
            objective (Optional[str]): The objective to evaluate against. Defaults to None.

        Returns:
            list[Score]: A list of Score objects, one for each supported piece in the request response.
        """
        if not request_response.request_pieces:
            return []

        # score the supported pieces
        supported_pieces = self._get_supported_pieces(request_response)

        tasks = [self._score_piece_async(request_piece=piece, objective=objective) for piece in supported_pieces]

        if not tasks:
            return []

        # Run all piece-level scorings concurrently
        piece_score_lists = await asyncio.gather(*tasks)

        # Flatten list[list[Score]] -> list[Score]
        return [score for sublist in piece_score_lists for score in sublist]

    async def _score_value_with_llm(
        self,
        *,
        prompt_target: PromptChatTarget,
        system_prompt: str,
        prompt_request_value: str,
        prompt_request_data_type: PromptDataType,
        scored_prompt_id: str | UUID,
        category: Optional[str | UUID] = None,
        objective: Optional[str] = None,
        score_value_output_key: str = "score_value",
        rationale_output_key: str = "rationale",
        description_output_key: str = "description",
        metadata_output_key: str = "metadata",
        category_output_key: str = "category",
        attack_identifier: Optional[Dict[str, str]] = None,
    ) -> UnvalidatedScore:
        score: UnvalidatedScore | None = None
        try:
            score = await super()._score_value_with_llm(
                prompt_target=prompt_target,
                system_prompt=system_prompt,
                prompt_request_value=prompt_request_value,
                prompt_request_data_type=prompt_request_data_type,
                scored_prompt_id=scored_prompt_id,
                category=category,
                objective=objective,
                score_value_output_key=score_value_output_key,
                rationale_output_key=rationale_output_key,
                description_output_key=description_output_key,
                metadata_output_key=metadata_output_key,
                category_output_key=category_output_key,
                attack_identifier=attack_identifier,
            )
            if score is None:
                raise ValueError("Score returned None")
            # raise an exception if it's not parsable as a float
            float(score.raw_score_value)
        except ValueError:
            score_value = score.raw_score_value if score else "None"
            raise InvalidJsonException(
                message=(f"Invalid JSON response, score_value should be a float not this: {score_value}")
            )
        return score
