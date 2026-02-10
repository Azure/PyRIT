# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter mappers – domain → DTO translation for converter-related models.
"""

from typing import Any

from pyrit.backend.models.converters import ConverterInstance


def converter_object_to_instance(converter_id: str, converter_obj: Any) -> ConverterInstance:
    """
    Build a ConverterInstance DTO from a registry converter object.

    Args:
        converter_id: The unique converter instance identifier.
        converter_obj: The domain PromptConverter object from the registry.

    Returns:
        ConverterInstance DTO with metadata derived from the object.
    """
    identifier = converter_obj.get_identifier()
    identifier_dict = identifier.to_dict()

    return ConverterInstance(
        converter_id=converter_id,
        type=identifier_dict.get("class_name", converter_obj.__class__.__name__),
        display_name=None,
        params=identifier_dict,
    )
