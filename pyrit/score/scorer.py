# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from abc import abstractmethod
import uuid

from pyrit.models import PromptRequestPiece
from pyrit.score import Score, ScoreType


class Scorer(abc.ABC):

    scorer_type: ScoreType

    @abstractmethod
    async def score(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Score the request_response, add the results to the database
        and return a list of Score objects.
        """
        raise NotImplementedError("score_text method not implemented")

    @abstractmethod
    def validate(self, request_response: PromptRequestPiece):
        """Validates the request_response piece to score"""
        raise NotImplementedError("score_text method not implemented")

    async def score_text(self, text: str) -> list[Score]:
        """
        Scores the given text using the chat target.
        """
        request_piece = PromptRequestPiece(
                    id=str(uuid.UUID(int=0)),
                    role="user",
                    original_value=text,
                )
        return await self.score(request_piece)

    def scale_value_float(self, value: float, min_value: float, max_value: float) -> float:
        # Scales a value from 0 to 1 from the min and max values
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_identifier(self):
        identifier = {}
        identifier["__type__"] = self.__class__.__name__
        identifier["__module__"] = self.__class__.__module__
        return identifier




