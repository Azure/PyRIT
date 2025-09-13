# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict, Iterable, List, TypeAlias, Union

from pyrit.common.utils import combine_dict
from pyrit.models import Score
from pyrit.score.score_aggregator_result import ScoreAggregatorResult


FloatScaleOp: TypeAlias = Callable[[List[float]], float]
FloatScaleScoreAggregator: TypeAlias = Callable[[Iterable[Score]], ScoreAggregatorResult]


def _lift(
    name: str,
    *,
    result_func: FloatScaleOp,
    aggregate_description: str,
) -> FloatScaleScoreAggregator:
    """Create a float-scale aggregator using a result function over float values.

    Args:
        name (str): Name of the aggregator variant.
        result_func (FloatScaleOp): Function applied to the list of float values to compute the aggregation result.
        aggregate_description (str): Base description for the aggregated result.

    Returns:
        FloatScaleScoreAggregator: Aggregator function that reduces a sequence of float-scale Scores
            into a ScoreAggregatorResult with a float value in [0, 1].
    """

    sep = "-"

    def aggregator(scores: Iterable[Score]) -> ScoreAggregatorResult:
        # Validate types and normalize input
        for s in scores:
            if s.score_type != "float_scale":
                raise ValueError("All scores must be of type 'float_scale'.")

        scores_list = list(scores)
        if not scores_list:
            # No scores; return a neutral result
            return ScoreAggregatorResult(
                value=0.0,
                description=f"No scores provided to {name} composite scorer.",
                rationale="",
                metadata={},
                category=[],
            )

        float_values = [float(s.get_value()) for s in scores_list]
        result = result_func(float_values)

        # Clamp result to [0, 1] defensively
        result = max(0.0, min(1.0, result))

        if len(scores_list) == 1:
            description = scores_list[0].score_value_description or ""
            rationale = scores_list[0].score_rationale or ""
        else:
            description=aggregate_description
            rationale = "\n".join(
                f"   {sep} {s.score_category}: {s.score_rationale or ''}" for s in scores_list
            )

        # Combine all score metadata dictionaries safely
        metadata: Dict[str, Union[str, int]] = {}
        category: List[str] = []
        for s in scores_list:
            metadata = combine_dict(metadata, getattr(s, "score_metadata", None))
            category.extend(getattr(s, "score_category", []) or [])


        return ScoreAggregatorResult(
            value=result,
            description=description,
            rationale=rationale,
            metadata=metadata,
            category=category,
        )

    aggregator.__name__ = f"{name}_"
    return aggregator


AVERAGE_ = _lift(
    "AVERAGE",
    result_func=lambda xs: sum(xs) / len(xs) if xs else 0.0,
    aggregate_description="Average of constituent scorers in an AVERAGE composite scorer.",
)

MAX_ = _lift(
    "MAX",
    result_func=max,
    aggregate_description="Maximum value among constituent scorers in a MAX composite scorer.",
)

MIN_ = _lift(
    "MIN",
    result_func=min,
    aggregate_description="Minimum value among constituent scorers in a MIN composite scorer.",
)
