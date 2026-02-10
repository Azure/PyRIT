# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target mappers – domain → DTO translation for target-related models.
"""

from typing import Any

from pyrit.backend.models.common import filter_sensitive_fields
from pyrit.backend.models.targets import TargetInstance


def target_object_to_instance(target_id: str, target_obj: Any) -> TargetInstance:
    """
    Build a TargetInstance DTO from a registry target object.

    Args:
        target_id: The unique target instance identifier.
        target_obj: The domain PromptTarget object from the registry.

    Returns:
        TargetInstance DTO with metadata derived from the object.
    """
    identifier = target_obj.get_identifier() if hasattr(target_obj, "get_identifier") else {}
    identifier_dict = identifier.to_dict() if hasattr(identifier, "to_dict") else identifier
    target_type = identifier_dict.get("__type__", target_obj.__class__.__name__)
    filtered_params = filter_sensitive_fields(identifier_dict)

    return TargetInstance(
        target_id=target_id,
        type=target_type,
        display_name=None,
        params=filtered_params,
    )
