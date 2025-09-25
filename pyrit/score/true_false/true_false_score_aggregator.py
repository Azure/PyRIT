# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import operator
from typing import Callable, Dict, Iterable, List, TypeAlias, Union

from pyrit.common.utils import combine_dict
from pyrit.models import Score
from pyrit.score.score_aggregator_result import ScoreAggregatorResult

BinaryBoolOp: TypeAlias = Callable[[bool, bool], bool]
TrueFalseScoreAggregator: TypeAlias = Callable[[Iterable[Score]], ScoreAggregatorResult]


def _lift(
    name: str,
    *,
    result_func: Callable[[List[bool]], bool],
    true_msg: str,
    false_msg: str,
) -> TrueFalseScoreAggregator:
    """Create a True/False aggregator using a result function over boolean values."""
    sep = "-"

    def aggregator(scores: Iterable[Score]) -> ScoreAggregatorResult:
        # Validate types and normalize input
        for s in scores:
            if s.score_type != "true_false":
                raise ValueError("All scores must be of type 'true_false'.")

        scores_list = list(scores)
        bool_values = [bool(s.get_value()) for s in scores_list]
        result = result_func(bool_values)

        # If there is only one score we're aggregating, use that. Else combine them
        # This makes scores more intuitive in many cases, where there is a single
        # text response, for example.
        if len(scores_list) == 1:
            description = scores_list[0].score_value_description or ""
            rationale = scores_list[0].score_rationale or ""
        else:
            description = true_msg if result else false_msg
            rationale = "\n".join(f"   {sep} {s.score_value}: {s.score_rationale or ''}" for s in scores_list)

        # Combine all score metadata dictionaries safely
        metadata: Dict[str, Union[str, int]] = {}
        category: List[str] = []
        for s in scores_list:
            metadata = combine_dict(metadata, getattr(s, "score_metadata", None))
            category.extend(getattr(s, "score_category", []))

        return ScoreAggregatorResult(
            value=result,
            description=description,
            rationale=rationale,
            metadata=metadata,
            category=category,
        )

    aggregator.__name__ = f"{name}_"
    return aggregator


def _lift_binary(
    name: str,
    op: BinaryBoolOp,
    true_msg: str,
    false_msg: str,
) -> TrueFalseScoreAggregator:
    """
    Turn a binary Boolean operator (e.g. operator.and_) into an aggregation function.
    """
    return _lift(
        name,
        result_func=lambda bs, _op=op: functools.reduce(_op, bs),  # type: ignore[misc]
        true_msg=true_msg,
        false_msg=false_msg,
    )


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

MAJORITY_ = _lift(
    "MAJORITY",
    result_func=lambda bs: sum(bs) > len(bs) / 2,
    true_msg="A strict majority of constituent scorers returned True in a MAJORITY composite scorer.",
    false_msg="A strict majority of constituent scorers did not return True in a MAJORITY composite scorer.",
)
