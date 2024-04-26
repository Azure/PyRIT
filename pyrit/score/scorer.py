# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from abc import abstractmethod

from pyrit.score import Score


class Scorer(abc.ABC):
    @abstractmethod
    def score_text(self, text: str) -> Score:
        """Score the text and return a Score object."""
        raise NotImplementedError("score_text method not implemented")
