# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, DefaultDict, Optional, Union

from pyrit.models import AttackOutcome, AttackResult

# ---------------------------------------------------------------------------
# Type alias for dimension extractors.
# An extractor receives an AttackResult and returns a list of string keys
# (list to support one-to-many mappings, e.g. multiple converters per attack).
# ---------------------------------------------------------------------------
DimensionExtractor = Callable[[AttackResult], list[str]]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AttackStats:
    """Statistics for attack analysis results."""

    success_rate: Optional[float]
    total_decided: int
    successes: int
    failures: int
    undetermined: int


@dataclass
class AnalysisResult:
    """
    Structured result from attack analysis.

    Attributes:
        overall (AttackStats): Aggregate stats across all attack results.
        dimensions (dict): Per-dimension breakdown. Keys are dimension names
            (str) for single dimensions, or tuples of dimension names for
            composite groupings. Values map dimension keys to AttackStats.
    """

    overall: AttackStats
    dimensions: dict[Union[str, tuple[str, ...]], dict[Union[str, tuple[str, ...]], AttackStats]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Built-in dimension extractors
# ---------------------------------------------------------------------------
def _extract_attack_type(result: AttackResult) -> list[str]:
    """
    Extract the attack type from the attack identifier dict.

    Reads the ``__type__`` key populated by :meth:`Strategy.get_identifier`.

    Returns:
        list[str]: A single-element list containing the attack type.
    """
    return [result.attack_identifier.get("__type__", "unknown")]


def _extract_converter_types(result: AttackResult) -> list[str]:
    """
    Extract converter class names from the last response.

    Returns:
        list[str]: Converter class names, or ``["no_converter"]`` if none.
    """
    if result.last_response is not None and result.last_response.converter_identifiers:
        return [conv.class_name for conv in result.last_response.converter_identifiers]
    return ["no_converter"]


def _extract_labels(result: AttackResult) -> list[str]:
    """
    Extract label key=value pairs from the last response.

    Returns:
        list[str]: Label strings as ``"key=value"``, or ``["no_labels"]`` if none.
    """
    if result.last_response is not None and result.last_response.labels:
        return [f"{k}={v}" for k, v in result.last_response.labels.items()]
    return ["no_labels"]


DEFAULT_DIMENSIONS: dict[str, DimensionExtractor] = {
    "attack_type": _extract_attack_type,
    "converter_type": _extract_converter_types,
    "label": _extract_labels,
}

# Deprecated aliases — maps old name to canonical name.
# Using the old name emits a DeprecationWarning.
_DEPRECATED_DIMENSION_ALIASES: dict[str, str] = {
    "attack_identifier": "attack_type",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_OUTCOME_KEYS: dict[AttackOutcome, str] = {
    AttackOutcome.SUCCESS: "successes",
    AttackOutcome.FAILURE: "failures",
}


def _outcome_key(outcome: AttackOutcome) -> str:
    """
    Map an AttackOutcome to its counter key.

    Returns:
        str: The counter key (``"successes"``, ``"failures"``, or ``"undetermined"``).
    """
    return _OUTCOME_KEYS.get(outcome, "undetermined")


def _compute_stats(*, successes: int, failures: int, undetermined: int) -> AttackStats:
    """
    Compute AttackStats from raw counts.

    Returns:
        AttackStats: The computed statistics.
    """
    total_decided = successes + failures
    success_rate = successes / total_decided if total_decided > 0 else None
    return AttackStats(
        success_rate=success_rate,
        total_decided=total_decided,
        successes=successes,
        failures=failures,
        undetermined=undetermined,
    )


def _build_stats(counts: DefaultDict[str, int]) -> AttackStats:
    """
    Build AttackStats from a counter dict.

    Returns:
        AttackStats: The computed statistics.
    """
    return _compute_stats(
        successes=counts["successes"],
        failures=counts["failures"],
        undetermined=counts["undetermined"],
    )


def _resolve_dimension_name(*, name: str, extractors: dict[str, DimensionExtractor]) -> str:
    """
    Resolve a single dimension name, handling deprecated aliases.

    Returns:
        str: The canonical dimension name.

    Raises:
        ValueError: If the dimension name is unknown.
    """
    if name in extractors:
        return name
    canonical = _DEPRECATED_DIMENSION_ALIASES.get(name)
    if canonical and canonical in extractors:
        warnings.warn(
            f"Dimension '{name}' is deprecated and will be removed in v0.13.0. Use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        return canonical
    raise ValueError(f"Unknown dimension '{name}'. Available: {sorted(extractors.keys())}")


def _resolve_dimension_spec(
    *, spec: Union[str, tuple[str, ...]], extractors: dict[str, DimensionExtractor]
) -> Union[str, tuple[str, ...]]:
    """
    Resolve a group_by spec (single or composite), handling deprecated aliases.

    Returns:
        Union[str, tuple[str, ...]]: The resolved spec with canonical dimension names.
    """
    if isinstance(spec, str):
        return _resolve_dimension_name(name=spec, extractors=extractors)
    return tuple(_resolve_dimension_name(name=n, extractors=extractors) for n in spec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_results(
    attack_results: list[AttackResult],
    *,
    group_by: list[Union[str, tuple[str, ...]]] | None = None,
    custom_dimensions: dict[str, DimensionExtractor] | None = None,
) -> AnalysisResult:
    """
    Analyze attack results with flexible, dimension-based grouping.

    Computes overall stats and breaks down results by one or more dimensions.
    Dimensions can be single (e.g. ``"converter_type"``) or composite tuples
    (e.g. ``("converter_type", "attack_type")``) for cross-dimensional
    grouping.

    Args:
        attack_results (list[AttackResult]): The attack results to analyze.
        group_by (list[str | tuple[str, ...]] | None): Dimensions to group by.
            Each element is either a dimension name (str) for independent
            grouping, or a tuple of dimension names for composite grouping.
            Defaults to all registered single dimensions.
        custom_dimensions (dict[str, DimensionExtractor] | None): Additional
            or overriding dimension extractors keyed by name. Merged with
            built-in defaults.

    Returns:
        AnalysisResult: Overall stats and per-dimension breakdowns.

    Raises:
        ValueError: If attack_results is empty or a dimension name is unknown.
        TypeError: If any element is not an AttackResult.

    Examples:
        Group by a single built-in dimension::

            result = analyze_results(attacks, group_by=["attack_type"])
            for name, stats in result.dimensions["attack_type"].items():
                print(f"{name}: {stats.success_rate}")

        Group by a composite (cross-product) of two dimensions::

            result = analyze_results(
                attacks,
                group_by=[("converter_type", "attack_type")],
            )

        Supply a custom dimension extractor::

            def by_objective(r: AttackResult) -> list[str]:
                return [r.objective]

            result = analyze_results(
                attacks,
                group_by=["objective"],
                custom_dimensions={"objective": by_objective},
            )
    """
    if not attack_results:
        raise ValueError("attack_results cannot be empty")

    # Merge extractors
    extractors = dict(DEFAULT_DIMENSIONS)
    if custom_dimensions:
        extractors.update(custom_dimensions)

    # Resolve group_by — default to every registered dimension independently
    if group_by is None:
        group_by = list(extractors.keys())

    # Resolve deprecated aliases and validate dimension names
    resolved_group_by: list[Union[str, tuple[str, ...]]] = []
    for spec in group_by:
        resolved_group_by.append(_resolve_dimension_spec(spec=spec, extractors=extractors))
    group_by = resolved_group_by

    # Accumulators
    overall_counts: DefaultDict[str, int] = defaultdict(int)
    dim_counts: dict[
        Union[str, tuple[str, ...]],
        DefaultDict[Union[str, tuple[str, ...]], DefaultDict[str, int]],
    ] = {spec: defaultdict(lambda: defaultdict(int)) for spec in group_by}

    # Single pass over results
    for attack in attack_results:
        if not isinstance(attack, AttackResult):
            raise TypeError(f"Expected AttackResult, got {type(attack).__name__}: {attack!r}")

        key = _outcome_key(attack.outcome)
        overall_counts[key] += 1

        for spec in group_by:
            if isinstance(spec, str):
                for dim_value in extractors[spec](attack):
                    dim_counts[spec][dim_value][key] += 1
            else:
                # Composite: cross-product of all sub-dimension values
                sub_values = [extractors[name](attack) for name in spec]
                for combo in product(*sub_values):
                    dim_counts[spec][combo][key] += 1

    # Build result
    dimension_stats: dict[Union[str, tuple[str, ...]], dict[Union[str, tuple[str, ...]], AttackStats]] = {}
    for spec, counts_by_key in dim_counts.items():
        dimension_stats[spec] = {dim_key: _build_stats(counts) for dim_key, counts in counts_by_key.items()}

    return AnalysisResult(
        overall=_build_stats(overall_counts),
        dimensions=dimension_stats,
    )
