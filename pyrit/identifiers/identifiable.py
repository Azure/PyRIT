# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from pyrit.identifiers.identifier import Identifier

# TypeVar bound to Identifier, allowing subclasses to specify their identifier type
IdentifierT = TypeVar("IdentifierT", bound=Identifier)


class Identifiable(ABC, Generic[IdentifierT]):
    """
    Abstract base class for objects that can provide a typed identifier.

    Generic over IdentifierT, allowing subclasses to specify their exact
    identifier type for strong typing support.

    Subclasses must implement `_build_identifier()` to construct their specific identifier.
    The `get_identifier()` method is provided and uses lazy building with caching.
    """

    _identifier: Optional[IdentifierT] = None

    @abstractmethod
    def _build_identifier(self) -> IdentifierT:
        """
        Build and return the identifier for this object.

        Subclasses must implement this method to construct their specific identifier type.
        This method is called lazily on first access via `get_identifier()`.

        Returns:
            IdentifierT: The constructed identifier for this component.
        """
        raise NotImplementedError("Subclasses must implement _build_identifier")

    def get_identifier(self) -> IdentifierT:
        """
        Get the typed identifier for this object. Built lazily on first access.

        Returns:
            IdentifierT: The identifier for this component.
        """
        if self._identifier is None:
            self._identifier = self._build_identifier()
        return self._identifier
