# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import operator
from dataclasses import dataclass
from typing import Callable, Iterable, TypeAlias

from pyrit.models import Score


@dataclass(frozen=True, slots=True)
class ScoreAggregatorResult:
    value: bool
    rationale: str


BinaryBoolOp: TypeAlias = Callable[[bool, bool], bool]
ScoreAggregator: TypeAlias = Callable[[Iterable[Score]], ScoreAggregatorResult]


def _lift_binary(
    name: str,
    op: BinaryBoolOp,
    true_msg: str,
    false_msg: str,
) -> ScoreAggregator:
    """
    Turn a binary Boolean operator (e.g. operator.and_) into an aggregation function
    """
    sep = "-"

    def aggregator(scores: Iterable[Score]) -> ScoreAggregatorResult:
        scores = list(scores)
        result = functools.reduce(op, (s.get_value() for s in scores))

        headline = true_msg if result else false_msg
        details = "\n".join(f"   {sep} {s.score_category}: {s.score_rationale or ''}" for s in scores)
        return ScoreAggregatorResult(value=result, rationale=f"{headline}\n{details}")

    aggregator.__name__ = f"{name}_"
    return aggregator


AND_ = _lift_binary(
    "AND",
    operator.and_,
    "All constituent scorers returned True in an AND composite scorer.",
    "At least one constituent scorer returned False in an AND composite scorer.",
)

OR_ = _lift_binary(
    "OR",
    operator.or_,
    "At least one constituent scorer returned True in an OR composite scorer.",
    "All constituent scorers returned False in an OR composite scorer.",
)


def MAJORITY_(scores: Iterable[Score]) -> ScoreAggregatorResult:
    scores = list(scores)
    result = sum(s.get_value() for s in scores) > len(scores) / 2
    headline = (
        "A strict majority of constituent scorers returned True."
        if result
        else "A strict majority of constituent scorers did not return True."
    )
    details = "\n".join(f"  - {s.score_category}: {s.score_rationale or ''}" for s in scores)
    return ScoreAggregatorResult(value=result, rationale=f"{headline}\n{details}")
