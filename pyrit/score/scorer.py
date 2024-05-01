# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from abc import abstractmethod
import json
from typing import Literal

from pyrit.models.literals import PromptDataType
from pyrit.models import PromptRequestPiece
from pyrit.score import Score
from pyrit.score.score_class import ScorerType



ScoreType = Literal["bool", "float_scale"]

class Scorer(abc.ABC):
    _score_type: ScorerType

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


    @abstractmethod
    def convert_value(self):
        """Converts the value to the appropriate type"""
        raise NotImplementedError("score_text method not implemented")


    def get_identifier(self):
        identifier = {}
        identifier["__type__"] = self.__class__.__name__
        identifier["__module__"] = self.__class__.__module__
        return identifier


class TrueFalseScorer(Scorer):

    def convert_value(self, score: Score) -> bool:
        return bool(score.score_value)

    def to_numeric_scorer(self, false_value = 0, true_value = 1) -> Scorer:
        # returns a LikertScorer based on the true threshhold
        raise NotImplementedError("to_numeric_scorer method not implemented")


class FloatScaleScorer(Scorer):
    score_type: ScorerType = "float_scale"

    def convert_value(self, score: Score) -> float:
        value = float(score.score_value)
        if value < 0 or value > 1:
            raise ValueError("Likert scale value must be between 0 and 1")

        return str(value)

    def scale_value(self, value: float, min_value: float, max_value: float) -> float:
        # Scales a value from 0 to 1 from the min and max values
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def to_true_false_scorer(self, true_threshold: float = .7) -> TrueFalseScorer:
        # returns a TrueFalseScorer based on the true threshhold
        raise NotImplementedError("to_true_false_scorer method not implemented")



