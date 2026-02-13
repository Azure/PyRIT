# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target mappers – domain → DTO translation for target-related models.
"""

from typing import Any

from pyrit.backend.models.targets import TargetInstance


def target_object_to_instance(target_unique_name: str, target_obj: Any) -> TargetInstance:
    """
    Build a TargetInstance DTO from a registry target object.

    Extracts only the frontend-relevant fields from the internal identifier,
    avoiding leakage of internal PyRIT core structures.

    Args:
        target_unique_name: The unique target instance identifier (registry key / unique_name).
        target_obj: The domain PromptTarget object from the registry.

    Returns:
        TargetInstance DTO with metadata derived from the object.
    """
    identifier = target_obj.get_identifier() if hasattr(target_obj, "get_identifier") else None

    return TargetInstance(
        target_unique_name=target_unique_name,
        target_type=identifier.class_name if identifier else target_obj.__class__.__name__,
        endpoint=getattr(identifier, "endpoint", None) if identifier else None,
        model_name=getattr(identifier, "model_name", None) if identifier else None,
        temperature=getattr(identifier, "temperature", None) if identifier else None,
        top_p=getattr(identifier, "top_p", None) if identifier else None,
        max_requests_per_minute=getattr(identifier, "max_requests_per_minute", None) if identifier else None,
        target_specific_params=getattr(identifier, "target_specific_params", None) if identifier else None,
    )
