# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter mappers – domain → DTO translation for converter-related models.
"""

from typing import Any, List, Optional

from pyrit.backend.models.converters import ConverterInstance


def converter_object_to_instance(
    converter_id: str,
    converter_obj: Any,
    *,
    sub_converter_ids: Optional[List[str]] = None,
) -> ConverterInstance:
    """
    Build a ConverterInstance DTO from a registry converter object.

    Extracts only the frontend-relevant fields from the internal identifier,
    avoiding leakage of internal PyRIT core structures.

    Args:
        converter_id: The unique converter instance identifier.
        converter_obj: The domain PromptConverter object from the registry.
        sub_converter_ids: Optional list of registered converter IDs for sub-converters.

    Returns:
        ConverterInstance DTO with metadata derived from the object.
    """
    identifier = converter_obj.get_identifier()

    return ConverterInstance(
        converter_id=converter_id,
        converter_type=identifier.class_name,
        display_name=None,
        supported_input_types=list(identifier.supported_input_types) if identifier.supported_input_types else [],
        supported_output_types=list(identifier.supported_output_types) if identifier.supported_output_types else [],
        converter_specific_params=identifier.converter_specific_params,
        sub_converter_ids=sub_converter_ids,
    )
