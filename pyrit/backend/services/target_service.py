# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target service for managing target instances.

Handles creation, retrieval, and lifecycle of runtime target instances.
Uses TargetRegistry as the source of truth.
"""

import importlib
import uuid
from typing import Any, List, Literal, Optional, cast

from pyrit.backend.models.common import filter_sensitive_fields
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)
from pyrit.registry.instance_registries import TargetRegistry


class TargetService:
    """
    Service for managing target instances.

    Uses TargetRegistry as the sole source of truth.
    API metadata is derived from the target objects' identifiers.
    """

    def __init__(self) -> None:
        """Initialize the target service."""
        self._registry = TargetRegistry.get_registry_singleton()

    def _get_target_class(self, target_type: str) -> type:
        """
        Get the target class for a given type.

        Returns:
            The target class matching the given type.
        """
        module = importlib.import_module("pyrit.prompt_target")

        cls = getattr(module, target_type, None)
        if cls is not None:
            return cast(type, cls)

        class_name_patterns = [
            target_type,
            f"{target_type}Target",
            "".join(word.capitalize() for word in target_type.split("_")),
            "".join(word.capitalize() for word in target_type.split("_")) + "Target",
        ]

        for pattern in class_name_patterns:
            cls = getattr(module, pattern, None)
            if cls is not None:
                return cast(type, cls)

        raise ValueError(f"Target type '{target_type}' not found in pyrit.prompt_target")

    def _build_instance_from_object(self, target_id: str, target_obj: Any) -> TargetInstance:
        """
        Build a TargetInstance from a registry object.

        Returns:
            TargetInstance with metadata derived from the object.
        """
        identifier = target_obj.get_identifier() if hasattr(target_obj, "get_identifier") else {}
        target_type = identifier.get("__type__", target_obj.__class__.__name__)
        filtered_params = filter_sensitive_fields(identifier)

        return TargetInstance(
            target_id=target_id,
            type=target_type,
            display_name=None,  # Could be added to identifier if needed
            params=filtered_params,
        )

    async def list_targets(
        self,
        source: Optional[Literal["initializer", "user"]] = None,
    ) -> TargetListResponse:
        """
        List all target instances.

        Returns:
            TargetListResponse containing all registered targets.
        """
        # source filter is ignored for now - all come from registry
        items: List[TargetInstance] = []
        for name in self._registry.get_names():
            obj = self._registry.get_instance_by_name(name)
            if obj:
                items.append(self._build_instance_from_object(name, obj))
        return TargetListResponse(items=items)

    async def get_target(self, target_id: str) -> Optional[TargetInstance]:
        """
        Get a target instance by ID.

        Returns:
            TargetInstance if found, None otherwise.
        """
        obj = self._registry.get_instance_by_name(target_id)
        if obj is None:
            return None
        return self._build_instance_from_object(target_id, obj)

    def get_target_object(self, target_id: str) -> Optional[Any]:
        """
        Get the actual target object for use in attacks.

        Returns:
            The PromptTarget object if found, None otherwise.
        """
        return self._registry.get_instance_by_name(target_id)

    async def create_target(
        self,
        request: CreateTargetRequest,
    ) -> CreateTargetResponse:
        """
        Create a new target instance.

        Returns:
            CreateTargetResponse with the new target's details.
        """
        target_id = str(uuid.uuid4())

        # Create and register the target object
        target_class = self._get_target_class(request.type)
        target_obj = target_class(**request.params)
        self._registry.register_instance(target_obj, name=target_id)

        # Build response from the object's identifier
        identifier = target_obj.get_identifier() if hasattr(target_obj, "get_identifier") else {}
        filtered_params = filter_sensitive_fields(identifier)

        return CreateTargetResponse(
            target_id=target_id,
            type=request.type,
            display_name=request.display_name,
            params=filtered_params,
        )


# Global service instance
_target_service: Optional[TargetService] = None


def get_target_service() -> TargetService:
    """
    Get the global target service instance.

    Returns:
        The singleton TargetService instance.
    """
    global _target_service
    if _target_service is None:
        _target_service = TargetService()
    return _target_service
