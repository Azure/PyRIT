# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from abc import abstractmethod

from pyrit.models import PromptRequestPiece
from pyrit.score import Score, ScoreType


class Scorer(abc.ABC):
    """
    Abstract base class for scorers.
    """

    scorer_type: ScoreType

    @abstractmethod
    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """
        Score the request_response, add the results to the database
        and return a list of Score objects.

        Args:
            request_response (PromptRequestPiece): The request response to be scored.

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        raise NotImplementedError("score_async method not implemented")

    @abstractmethod
    def validate(self, request_response: PromptRequestPiece):
        """
        Validates the request_response piece to score. Because some scorers may require
        specific PromptRequestPiece types or values.

        Args:
            request_response (PromptRequestPiece): The request response to be validated.
        """
        raise NotImplementedError("score_async method not implemented")

    async def score_text_async(self, text: str) -> list[Score]:
        """
        Scores the given text using the chat target.

        Args:
            text (str): The text to be scored.

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        request_piece = PromptRequestPiece(
            role="user",
            original_value=text,
        )

        request_piece.id = None
        return await self.score_async(request_piece)

    async def score_image_async(self, image_path: str) -> list[Score]:
        """
        Scores the given image using the chat target.

        Args:
            text (str): The image to be scored.

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        request_piece = PromptRequestPiece(
            role="user",
            original_value=image_path,
            converted_value=image_path,
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
        )

        request_piece.id = None
        return await self.score_async(request_piece)

    def scale_value_float(self, value: float, min_value: float, max_value: float) -> float:
        """
        Scales a value from 0 to 1 based on the given min and max values. E.g. 3 stars out of 5 stars would be .5.

        Args:
            value (float): The value to be scaled.
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.

        Returns:
            float: The scaled value.
        """
        if max_value == min_value:
            return 0.0

        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_identifier(self):
        """
        Returns an identifier dictionary for the scorer.

        Returns:
            dict: The identifier dictionary.
        """
        identifier = {}
        identifier["__type__"] = self.__class__.__name__
        identifier["__module__"] = self.__class__.__module__
        return identifier
