# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Krippendorff's alpha for ordinal data.

This module exposes an ordinal-only implementation of Krippendorff's alpha used by
the harm scorer evaluation.
"""

from __future__ import annotations

import numpy as np


def alpha(
    reliability_data: "np.ndarray",  # shape: (num_raters_or_trials, num_items); dtype float
    level_of_measurement: str = "ordinal",
    missing: float | None = np.nan,
) -> float:
    """Compute Krippendorff's alpha inter-rater reliability for ordinal data.

    Computes inter-rater reliability for ordered categories, ignoring missing
    entries and supporting varying numbers of raters per item.

    Args:
        reliability_data (np.ndarray):
            Ratings array of shape (num_raters_or_trials, num_items). Each entry
            is a numeric rating on an ordered scale. Missing ratings should be
            represented by ``np.nan`` (default) or the value specified by
            ``missing``.
        level_of_measurement (str):
            Level of measurement. Must be ``"ordinal"``. Any other value will
            raise ``ValueError``.
        missing (float | None):
            Sentinel value indicating missing ratings. If ``None``, it is
            treated as ``np.nan``. Defaults to ``np.nan``.

    Returns:
        float: Krippendorff's alpha, where 1.0 indicates perfect agreement. The
        value can be below 0 when agreement is worse than chance. ``np.nan`` is
        returned when the statistic is undefined (e.g., fewer than two usable
        ratings or zero expected disagreement with non-zero observed
        disagreement).

    Raises:
        ValueError: If ``level_of_measurement`` is not ``"ordinal"``.
    """
    # Validate level of measurement
    if level_of_measurement != "ordinal":
        raise ValueError(f"Only 'ordinal' level of measurement is supported, got '{level_of_measurement}'")

    # Convert to float64 for numerical precision
    data = np.asarray(reliability_data, dtype=np.float64)

    # Handle missing value indicator
    if missing is None:
        missing = np.nan

    # Identify valid (non-missing) values
    if np.isnan(missing):
        valid_mask = ~np.isnan(data)
    else:
        valid_mask = data != missing

    # Extract valid ratings and get unique categories
    valid_ratings = data[valid_mask]

    # Check if we have enough data
    if len(valid_ratings) < 2:
        return np.nan

    # Get sorted unique categories
    categories = np.unique(valid_ratings)
    num_categories = len(categories)

    # All ratings identical - perfect agreement
    if num_categories == 1:
        return 1.0

    # Build value counts matrix (items x categories)
    # This represents how many raters assigned each category to each item
    num_items = data.shape[1]
    value_counts = np.zeros((num_items, num_categories), dtype=np.int64)

    for item_idx in range(num_items):
        item_ratings = data[:, item_idx]
        item_valid = valid_mask[:, item_idx]
        valid_item_ratings = item_ratings[item_valid]

        for rating in valid_item_ratings:
            cat_idx = np.searchsorted(categories, rating)
            value_counts[item_idx, cat_idx] += 1

    # Calculate pairable values per item (must have at least 2 raters)
    pairable = np.maximum(value_counts.sum(axis=1), 2)

    # Build coincidence matrix using vectorized operations
    coincidence_matrix = np.zeros((num_categories, num_categories), dtype=np.float64)

    for item_idx in range(num_items):
        item_counts = value_counts[item_idx]
        m_c = pairable[item_idx]

        if m_c < 2:
            continue

        # Compute outer product for this item
        item_coincidences = np.outer(item_counts, item_counts).astype(np.float64)

        # Set diagonal to n_i * (n_i - 1) to remove self-pairs
        np.fill_diagonal(item_coincidences, item_counts * (item_counts - 1))

        # Normalize and add to total
        coincidence_matrix += item_coincidences / (m_c - 1)

    # Calculate marginals (total coincidences per category)
    n_v = coincidence_matrix.sum(axis=0)
    total_n = n_v.sum()

    # Check for degenerate case
    if total_n == 0:
        return np.nan

    # Expected coincidence matrix
    expected_matrix = np.outer(n_v, n_v) - np.diag(n_v)
    expected_matrix = expected_matrix / (total_n - 1)

    # Ordinal distance matrix using category marginals
    distance_matrix = np.zeros((num_categories, num_categories), dtype=np.float64)
    for i in range(num_categories):
        for j in range(num_categories):
            if i == j:
                continue
            min_idx, max_idx = (i, j) if i < j else (j, i)
            between_sum = n_v[min_idx : max_idx + 1].sum() - (n_v[min_idx] + n_v[max_idx]) / 2.0
            distance_matrix[i, j] = between_sum**2

    # Observed and expected disagreements
    observed_disagreement = float(np.sum(coincidence_matrix * distance_matrix))
    expected_disagreement = float(np.sum(expected_matrix * distance_matrix))

    # Handle edge case where expected disagreement is zero
    if np.abs(expected_disagreement) < 1e-15:
        if np.abs(observed_disagreement) < 1e-15:
            return 1.0
        return np.nan

    # Krippendorff's alpha
    alpha_value = 1.0 - observed_disagreement / expected_disagreement

    # Clip tiny negative values due to rounding
    if -1e-10 < alpha_value < 0:
        alpha_value = 0.0

    return float(alpha_value)
