# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Labels API routes.

Provides access to unique label values for filtering in the GUI.
"""

from typing import TYPE_CHECKING, Dict, List, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from pyrit.memory import CentralMemory

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface

router = APIRouter(prefix="/labels", tags=["labels"])


class LabelOptionsResponse(BaseModel):
    """Response containing unique label keys and their values."""

    source: str = Field(..., description="Source type (e.g., 'attacks')")
    labels: Dict[str, List[str]] = Field(..., description="Map of label keys to their unique values")


@router.get(
    "",
    response_model=LabelOptionsResponse,
)
async def get_label_options(
    source: Literal["attacks"] = Query(
        "attacks",
        description="Source type to get labels from. Currently only 'attacks' is supported.",
    ),
) -> LabelOptionsResponse:
    """
    Get unique label keys and values for filtering.

    Returns all unique label key-value combinations from the specified source.
    Useful for populating filter dropdowns in the GUI.

    Args:
        source: The source type to query labels from.

    Returns:
        LabelOptionsResponse: Map of label keys to their unique values.
    """
    memory = CentralMemory.get_memory_instance()

    if source == "attacks":
        labels = _get_attack_labels(memory)
    else:
        # Future: add support for other sources
        labels = {}

    return LabelOptionsResponse(source=source, labels=labels)


def _get_attack_labels(memory: "MemoryInterface") -> Dict[str, List[str]]:
    """
    Extract unique labels from all attack results.

    Returns:
        Dict mapping label keys to sorted lists of unique values.
    """
    attack_results = memory.get_attack_results()

    # Collect all unique key-value pairs
    label_values: Dict[str, set[str]] = {}

    for ar in attack_results:
        if ar.metadata:
            for key, value in ar.metadata.items():
                # Skip internal metadata keys
                if key.startswith("_") or key in ("created_at", "updated_at"):
                    continue
                # Only include string values
                if isinstance(value, str):
                    if key not in label_values:
                        label_values[key] = set()
                    label_values[key].add(value)

    # Convert sets to sorted lists
    return {key: sorted(values) for key, values in sorted(label_values.items())}
