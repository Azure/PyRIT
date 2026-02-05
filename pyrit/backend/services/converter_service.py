# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter service for managing converter instances.

Handles creation, retrieval, and preview of converters.
Uses ConverterRegistry as the source of truth.

If a converter requires another converter (e.g., SelectiveTextConverter),
the inner converter must be created first and passed by ID in params.
"""

import importlib
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterInstanceListResponse,
    ConverterPreviewRequest,
    ConverterPreviewResponse,
    CreateConverterRequest,
    CreateConverterResponse,
    PreviewStep,
)
from pyrit.models import PromptDataType
from pyrit.registry.instance_registries import ConverterRegistry


class ConverterService:
    """
    Service for managing converter instances.

    Uses ConverterRegistry as the sole source of truth.
    API metadata is derived from the converter objects.
    """

    def __init__(self) -> None:
        """Initialize the converter service."""
        self._registry = ConverterRegistry.get_registry_singleton()

    def _build_instance_from_object(self, converter_id: str, converter_obj: Any) -> ConverterInstance:
        """
        Build a ConverterInstance from a registry object.

        Returns:
            ConverterInstance with metadata derived from the object.
        """
        converter_type = converter_obj.__class__.__name__
        return ConverterInstance(
            converter_id=converter_id,
            type=converter_type,
            display_name=None,
            params={},  # Params aren't stored on converter objects
        )

    # ========================================================================
    # Public API Methods
    # ========================================================================

    async def list_converters(
        self, source: Optional[Literal["initializer", "user"]] = None
    ) -> ConverterInstanceListResponse:
        """
        List all converter instances.

        Returns:
            ConverterInstanceListResponse containing all registered converters.
        """
        # source filter is ignored for now - all come from registry
        items: List[ConverterInstance] = []
        for name in self._registry.get_names():
            obj = self._registry.get_instance_by_name(name)
            if obj:
                items.append(self._build_instance_from_object(name, obj))
        return ConverterInstanceListResponse(items=items)

    async def get_converter(self, converter_id: str) -> Optional[ConverterInstance]:
        """
        Get a converter instance by ID.

        Returns:
            ConverterInstance if found, None otherwise.
        """
        obj = self._registry.get_instance_by_name(converter_id)
        if obj is None:
            return None
        return self._build_instance_from_object(converter_id, obj)

    def get_converter_object(self, converter_id: str) -> Optional[Any]:
        """
        Get the actual converter object.

        Returns:
            The PromptConverter object if found, None otherwise.
        """
        return self._registry.get_instance_by_name(converter_id)

    async def create_converter(self, request: CreateConverterRequest) -> CreateConverterResponse:
        """
        Create a new converter instance.

        If params contains a 'converter' key with a converter_id,
        the referenced converter object will be resolved and passed.

        Returns:
            CreateConverterResponse with the new converter's details.
        """
        converter_id = str(uuid.uuid4())

        # Resolve any converter references in params and create the object
        params = self._resolve_converter_params(request.params)
        converter_obj = self._get_converter_class(request.type)(**params)
        self._registry.register_instance(converter_obj, name=converter_id)

        return CreateConverterResponse(
            converter_id=converter_id,
            type=request.type,
            display_name=request.display_name,
            params=request.params,
        )

    async def preview_conversion(self, request: ConverterPreviewRequest) -> ConverterPreviewResponse:
        """
        Preview conversion through a converter pipeline.

        Returns:
            ConverterPreviewResponse with step-by-step conversion results.
        """
        converters = self._gather_converters(request.converter_ids)
        steps, final_value, final_type = await self._apply_converters(
            converters, request.original_value, request.original_value_data_type
        )

        return ConverterPreviewResponse(
            original_value=request.original_value,
            original_value_data_type=request.original_value_data_type,
            converted_value=final_value,
            converted_value_data_type=final_type,
            steps=steps,
        )

    def get_converter_objects_for_ids(self, converter_ids: List[str]) -> List[Any]:
        """
        Get converter objects for a list of IDs.

        Returns:
            List of converter objects in the same order as the input IDs.
        """
        converters = []
        for conv_id in converter_ids:
            conv_obj = self.get_converter_object(conv_id)
            if conv_obj is None:
                raise ValueError(f"Converter instance '{conv_id}' not found")
            converters.append(conv_obj)
        return converters

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _resolve_converter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve converter references in params.

        Returns:
            Params dict with converter_id references replaced by actual objects.
        """
        resolved = dict(params)
        if "converter" in resolved and isinstance(resolved["converter"], dict):
            ref = resolved["converter"]
            if "converter_id" in ref:
                conv_obj = self.get_converter_object(ref["converter_id"])
                if conv_obj is None:
                    raise ValueError(f"Referenced converter '{ref['converter_id']}' not found")
                resolved["converter"] = conv_obj
        return resolved

    def _get_converter_class(self, converter_type: str) -> type:
        """
        Get the converter class for a given type.

        Returns:
            The converter class matching the given type.
        """
        module = importlib.import_module("pyrit.prompt_converter")

        cls = getattr(module, converter_type, None)
        if cls is not None:
            return cast(type, cls)

        for pattern in self._class_name_patterns(converter_type):
            cls = getattr(module, pattern, None)
            if cls is not None:
                return cast(type, cls)

        raise ValueError(f"Converter type '{converter_type}' not found in pyrit.prompt_converter")

    def _class_name_patterns(self, type_name: str) -> List[str]:
        """
        Generate class name patterns to try.

        Returns:
            List of possible class name variations.
        """
        pascal = "".join(word.capitalize() for word in type_name.split("_"))
        return [type_name, f"{type_name}Converter", pascal, f"{pascal}Converter"]

    def _gather_converters(self, converter_ids: List[str]) -> List[Tuple[str, str, Any]]:
        """
        Gather converters to apply from IDs.

        Returns:
            List of tuples (converter_id, converter_type, converter_obj).
        """
        converters: List[Tuple[str, str, Any]] = []
        for conv_id in converter_ids:
            conv_obj = self.get_converter_object(conv_id)
            if conv_obj is None:
                raise ValueError(f"Converter instance '{conv_id}' not found")
            conv_type = conv_obj.__class__.__name__
            converters.append((conv_id, conv_type, conv_obj))
        return converters

    async def _apply_converters(
        self,
        converters: List[Tuple[str, str, Any]],
        initial_value: str,
        initial_type: PromptDataType,
    ) -> Tuple[List[PreviewStep], str, PromptDataType]:
        """
        Apply converters and collect steps.

        Returns:
            Tuple of (steps, final_value, final_type).
        """
        current_value = initial_value
        current_type = initial_type
        steps: List[PreviewStep] = []

        for conv_id, conv_type, conv_obj in converters:
            input_value, input_type = current_value, current_type
            result = await conv_obj.convert_async(prompt=current_value)
            current_value, current_type = result.output_text, result.output_type

            steps.append(
                PreviewStep(
                    converter_id=conv_id,
                    converter_type=conv_type,
                    input_value=input_value,
                    input_data_type=input_type,
                    output_value=current_value,
                    output_data_type=current_type,
                )
            )

        return steps, current_value, current_type


# ============================================================================
# Singleton
# ============================================================================

_converter_service: Optional[ConverterService] = None


def get_converter_service() -> ConverterService:
    """
    Get the global converter service instance.

    Returns:
        The singleton ConverterService instance.
    """
    global _converter_service
    if _converter_service is None:
        _converter_service = ConverterService()
    return _converter_service
