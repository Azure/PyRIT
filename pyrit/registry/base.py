# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared base types for PyRIT registries.

This module contains types shared between class registries (which store Type[T])
and instance registries (which store T instances).
"""

from dataclasses import dataclass
from typing import Any, Iterator, List, Protocol, TypeVar, runtime_checkable

# Type variable for metadata (invariant for Protocol compatibility)
MetadataT = TypeVar("MetadataT")


def _matches_filters(metadata: Any, **filters: Any) -> bool:
    """
    Check if a metadata object matches all provided filters.

    Supports filtering on any property of the metadata dataclass:
    - For simple types (str, int, bool): exact match comparison
    - For sequence types (list, tuple): checks if filter value is contained in the sequence

    Args:
        metadata: The metadata dataclass instance to check.
        **filters: Keyword arguments where key is the property name and value is the filter.

    Returns:
        True if all filters match, False otherwise.
    """
    for key, filter_value in filters.items():
        if not hasattr(metadata, key):
            return False

        actual_value = getattr(metadata, key)

        # Handle sequence types - check if filter value is in the sequence
        if isinstance(actual_value, (list, tuple)):
            if filter_value not in actual_value:
                return False
        # Simple exact match for other types
        elif actual_value != filter_value:
            return False

    return True


@runtime_checkable
class RegistryProtocol(Protocol[MetadataT]):
    """
    Protocol defining the common interface for all registries.

    Both class registries (BaseClassRegistry) and instance registries
    (BaseInstanceRegistry) implement this interface, enabling code that
    works with either registry type.

    Type Parameters:
        MetadataT: The metadata dataclass type (e.g., ScenarioMetadata).
    """

    @classmethod
    def get_instance(cls) -> "RegistryProtocol[MetadataT]":
        """Get the singleton instance of this registry."""
        ...

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        ...

    def get_names(self) -> List[str]:
        """Get a sorted list of all registered names."""
        ...

    def list_metadata(self, **filters: Any) -> List[MetadataT]:
        """List metadata for all registered items, optionally filtered."""
        ...

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        ...

    def __len__(self) -> int:
        """Get the count of registered items."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        ...


@dataclass(frozen=True)
class RegistryItemMetadata:
    """
    Base dataclass for registry item metadata.

    This dataclass provides descriptive information about a registered item
    (either a class or an instance). It is NOT the item itself - it's a
    structured object describing the item.

    All registry-specific metadata types should extend this with additional fields.
    """

    name: str  # The snake_case registry name (e.g., "self_ask_refusal")
    class_name: str  # The actual class name (e.g., "SelfAskRefusalScorer")
    description: str  # Description from docstring or manual override
