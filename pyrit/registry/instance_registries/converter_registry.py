# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter registry for managing PyRIT converter instances.

Converters are registered explicitly via initializers as pre-configured instances.

NOTE: This is a placeholder implementation. A full implementation will be added soon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pyrit.identifiers import Identifier
from pyrit.identifiers.class_name_utils import class_name_to_snake_case
from pyrit.registry.instance_registries.base_instance_registry import (
    BaseInstanceRegistry,
)

if TYPE_CHECKING:
    from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


# Placeholder identifier type until proper ConverterIdentifier is defined
# TODO: Replace with ConverterIdentifier when available
@dataclass(frozen=True)
class ConverterIdentifier(Identifier):
    """Temporary identifier type for converters."""

    pass


class ConverterRegistry(BaseInstanceRegistry["PromptConverter", ConverterIdentifier]):
    """
    Registry for managing available converter instances.

    This registry stores pre-configured PromptConverter instances (not classes).
    Converters are registered explicitly via initializers after being instantiated
    with their required parameters.

    NOTE: This is a placeholder. A full implementation will be added soon.
    """

    @classmethod
    def get_registry_singleton(cls) -> "ConverterRegistry":
        """
        Get the singleton instance of the ConverterRegistry.

        Returns:
            The singleton ConverterRegistry instance.
        """
        return super().get_registry_singleton()  # type: ignore[return-value]

    def register_instance(
        self,
        converter: "PromptConverter",
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a converter instance.

        Args:
            converter: The pre-configured converter instance (not a class).
            name: Optional custom registry name. If not provided,
                derived from class name (e.g., Base64Converter -> base64).
        """
        if name is None:
            name = class_name_to_snake_case(converter.__class__.__name__, suffix="Converter")

        self.register(converter, name=name)
        logger.debug(f"Registered converter instance: {name} ({converter.__class__.__name__})")

    def get_instance_by_name(self, name: str) -> Optional["PromptConverter"]:
        """
        Get a registered converter instance by name.

        Args:
            name: The registry name of the converter.

        Returns:
            The converter instance, or None if not found.
        """
        return self.get(name)

    def _build_metadata(self, name: str, instance: "PromptConverter") -> ConverterIdentifier:
        """
        Build metadata for a converter instance.

        Args:
            name: The registry name of the converter.
            instance: The converter instance.

        Returns:
            ConverterIdentifier with basic info about the converter.
        """
        return ConverterIdentifier(
            class_name=instance.__class__.__name__,
            class_module=instance.__class__.__module__,
            class_description=f"Converter: {name}",
            identifier_type="instance",
        )
