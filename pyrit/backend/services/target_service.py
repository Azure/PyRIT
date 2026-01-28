# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target service for managing target instances.

Handles creation, retrieval, and lifecycle of runtime target instances.
"""

import importlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, cast

from pyrit.backend.models.common import filter_sensitive_fields
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)


class TargetService:
    """Service for managing target instances."""

    def __init__(self) -> None:
        """Initialize the target service."""
        # In-memory storage for target instances
        self._instances: Dict[str, TargetInstance] = {}
        # Actual instantiated target objects (not serializable)
        self._target_objects: Dict[str, Any] = {}

    def _get_target_class(self, target_type: str) -> type:
        """
        Get the target class for a given type.

        Args:
            target_type: Target type string (e.g., 'azure_openai', 'TextTarget')

        Returns:
            The target class
        """
        # Try to import from pyrit.prompt_target
        module = importlib.import_module("pyrit.prompt_target")

        # Handle both snake_case and PascalCase
        # First try direct attribute lookup
        cls = getattr(module, target_type, None)
        if cls is not None:
            return cast(type, cls)

        # Try common class name patterns
        class_name_patterns = [
            target_type,
            f"{target_type}Target",
            "".join(word.capitalize() for word in target_type.split("_")),  # snake_case to PascalCase
            "".join(word.capitalize() for word in target_type.split("_")) + "Target",
        ]

        for pattern in class_name_patterns:
            cls = getattr(module, pattern, None)
            if cls is not None:
                return cast(type, cls)

        raise ValueError(f"Target type '{target_type}' not found in pyrit.prompt_target")

    async def list_targets(
        self,
        source: Optional[Literal["initializer", "user"]] = None,
    ) -> TargetListResponse:
        """
        List all target instances.

        Args:
            source: Optional filter by source ("initializer" or "user")

        Returns:
            TargetListResponse: List of target instances
        """
        items = list(self._instances.values())

        if source is not None:
            items = [t for t in items if t.source == source]

        return TargetListResponse(items=items)

    async def get_target(self, target_id: str) -> Optional[TargetInstance]:
        """
        Get a target instance by ID.

        Args:
            target_id: Target instance ID

        Returns:
            TargetInstance or None if not found
        """
        return self._instances.get(target_id)

    def get_target_object(self, target_id: str) -> Optional[Any]:
        """
        Get the actual target object for use in attacks.

        Args:
            target_id: Target instance ID

        Returns:
            The instantiated target object or None if not found
        """
        return self._target_objects.get(target_id)

    async def create_target(
        self,
        request: CreateTargetRequest,
    ) -> CreateTargetResponse:
        """
        Create a new target instance.

        Args:
            request: Target creation request

        Returns:
            CreateTargetResponse: Created target details
        """
        target_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Get the target class and instantiate
        target_class = self._get_target_class(request.type)
        target_obj = target_class(**request.params)
        self._target_objects[target_id] = target_obj

        # Get filtered params from target identifier
        target_identifier = target_obj.get_identifier() if hasattr(target_obj, "get_identifier") else {}
        filtered_params = filter_sensitive_fields(target_identifier)

        # Store the target instance metadata
        instance = TargetInstance(
            target_id=target_id,
            type=request.type,
            display_name=request.display_name,
            params=filtered_params,
            created_at=now,
            source="user",
        )
        self._instances[target_id] = instance

        return CreateTargetResponse(
            target_id=target_id,
            type=request.type,
            display_name=request.display_name,
            params=filtered_params,
            created_at=now,
            source="user",
        )

    async def delete_target(self, target_id: str) -> bool:
        """
        Delete a target instance.

        Args:
            target_id: Target instance ID

        Returns:
            True if deleted, False if not found
        """
        if target_id in self._instances:
            del self._instances[target_id]
            self._target_objects.pop(target_id, None)
            return True
        return False

    async def register_initializer_target(
        self,
        target_type: str,
        target_obj: Any,
        display_name: Optional[str] = None,
    ) -> TargetInstance:
        """
        Register a target from an initializer (not user-created).

        Args:
            target_type: Target type string
            target_obj: Already-instantiated target object
            display_name: Optional display name

        Returns:
            TargetInstance: The registered target
        """
        target_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Store the target object
        self._target_objects[target_id] = target_obj

        # Get filtered params from target identifier
        target_identifier = target_obj.get_identifier() if hasattr(target_obj, "get_identifier") else {}
        filtered_params = filter_sensitive_fields(target_identifier)

        instance = TargetInstance(
            target_id=target_id,
            type=target_type,
            display_name=display_name,
            params=filtered_params,
            created_at=now,
            source="initializer",
        )
        self._instances[target_id] = instance

        return instance


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
