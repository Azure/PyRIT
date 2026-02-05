# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target service for managing target instances.

Handles creation and retrieval of target instances.
Uses TargetRegistry as the source of truth for instances.

Targets can be:
- Created via API request (instantiated from request params, then registered)
- Retrieved from registry (pre-registered at startup or created earlier)
"""

import uuid
from typing import Any, Optional

from pyrit import prompt_target
from pyrit.backend.models.common import filter_sensitive_fields
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)
from pyrit.prompt_target import PromptTarget
from pyrit.registry.instance_registries import TargetRegistry


def _build_target_class_registry() -> dict[str, type]:
    """
    Build a registry mapping target class names to their classes.

    Uses the prompt_target module's __all__ to discover all available targets.

    Returns:
        Dict mapping class name (str) to class (type).
    """
    registry: dict[str, type] = {}
    for name in prompt_target.__all__:
        cls = getattr(prompt_target, name, None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, PromptTarget):
            registry[name] = cls
    return registry


# Module-level class registry (built once on import)
_TARGET_CLASS_REGISTRY: dict[str, type] = _build_target_class_registry()


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
        Get the target class for a given type name.

        Looks up the class in the module-level target class registry.

        Args:
            target_type: The exact class name of the target (e.g., 'TextTarget').

        Returns:
            The target class.

        Raises:
            ValueError: If the target type is not found.
        """
        cls = _TARGET_CLASS_REGISTRY.get(target_type)
        if cls is None:
            raise ValueError(
                f"Target type '{target_type}' not found. Available types: {sorted(_TARGET_CLASS_REGISTRY.keys())}"
            )
        return cls

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

    async def list_targets(self) -> TargetListResponse:
        """
        List all target instances.

        Returns:
            TargetListResponse containing all registered targets.
        """
        items = [
            self._build_instance_from_object(name, obj) for name, obj in self._registry.get_all_instances().items()
        ]
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

    async def create_target(self, request: CreateTargetRequest) -> CreateTargetResponse:
        """
        Create a new target instance from API request.

        Instantiates the target with the given type and params,
        then registers it in the registry.

        Args:
            request: The create target request with type and params.

        Returns:
            CreateTargetResponse with the new target's details.

        Raises:
            ValueError: If the target type is not found.
        """
        target_id = str(uuid.uuid4())

        # Instantiate from request params and register
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
