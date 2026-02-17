# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target mappers – domain → DTO translation for target-related models.
"""

from pyrit.backend.models.targets import TargetInstance
from pyrit.prompt_target import PromptTarget


def target_object_to_instance(target_registry_name: str, target_obj: PromptTarget) -> TargetInstance:
    """
    Build a TargetInstance DTO from a registry target object.

    Extracts only the frontend-relevant fields from the internal identifier,
    avoiding leakage of internal PyRIT core structures.

    Args:
        target_registry_name: The human-friendly target registry name.
        target_obj: The domain PromptTarget object from the registry.

    Returns:
        TargetInstance DTO with metadata derived from the object.
    """
    identifier = target_obj.get_identifier()

    return TargetInstance(
        target_registry_name=target_registry_name,
        target_type=identifier.class_name,
        endpoint=identifier.endpoint or None,
        model_name=identifier.model_name or None,
        temperature=identifier.temperature,
        top_p=identifier.top_p,
        max_requests_per_minute=identifier.max_requests_per_minute,
        target_specific_params=identifier.target_specific_params,
    )
