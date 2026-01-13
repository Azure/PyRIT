# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Dict, List, Union

from pyrit.common.utils import combine_dict
from pyrit.models import Score


def combine_metadata_and_categories(scores: List[Score]) -> tuple[Dict[str, Union[str, int]], List[str]]:
    """
    Combine metadata and categories from multiple scores with deduplication.

    Args:
        scores: List of Score objects.

    Returns:
        Tuple of (metadata dict, sorted category list with empty strings filtered).
    """
    metadata: Dict[str, Union[str, int]] = {}
    category_set: set[str] = set()

    for s in scores:
        metadata = combine_dict(metadata, getattr(s, "score_metadata", None))
        score_categories = getattr(s, "score_category", None) or []
        category_set.update([c for c in score_categories if c])

    category = sorted(category_set)
    return metadata, category


def format_score_for_rationale(score: Score) -> str:
    """
    Format a single score for inclusion in an aggregated rationale.

    Args:
        score: The Score object to format.

    Returns:
        Formatted string with scorer class, value, and rationale.
    """
    class_type = score.scorer_class_identifier.get("__type__", "Unknown")
    return f"   - {class_type} {score.score_value}: {score.score_rationale or ''}"
