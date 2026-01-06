# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared base types for PyRIT registries.

This module contains types shared between class registries (which store Type[T])
and instance registries (which store T instances).
"""

from typing import Iterator, List, Protocol, TypedDict, TypeVar, runtime_checkable

# Type variable for metadata (invariant for Protocol compatibility)
MetadataT = TypeVar("MetadataT")


@runtime_checkable
class RegistryProtocol(Protocol[MetadataT]):
    """
    Protocol defining the common interface for all registries.

    Both class registries (BaseClassRegistry) and instance registries
    (BaseInstanceRegistry) implement this interface, enabling code that
    works with either registry type.

    Type Parameters:
        MetadataT: The TypedDict type for metadata (e.g., ScenarioMetadata).
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

    def list_metadata(self) -> List[MetadataT]:
        """List metadata for all registered items."""
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


class RegistryItemMetadata(TypedDict):
    """
    Base type definition for registry item metadata.

    This TypedDict provides descriptive information about a registered item
    (either a class or an instance). It is NOT the item itself - it's a
    dictionary describing the item.

    All registry-specific metadata types should extend this with additional fields.
    """

    name: str  # The snake_case registry name (e.g., "self_ask_refusal")
    class_name: str  # The actual class name (e.g., "SelfAskRefusalScorer")
    description: str  # Description from docstring or manual override
