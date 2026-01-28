# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter service for managing converter instances.

Handles creation, retrieval, and nested converter support.
"""

import importlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterInstanceListResponse,
    ConverterPreviewRequest,
    ConverterPreviewResponse,
    CreateConverterRequest,
    CreateConverterResponse,
    InlineConverterConfig,
    PreviewStep,
)
from pyrit.models import PromptDataType


class ConverterService:
    """Service for managing converter instances."""

    def __init__(self) -> None:
        """Initialize the converter service."""
        self._instances: Dict[str, ConverterInstance] = {}
        self._converter_objects: Dict[str, Any] = {}

    # ========================================================================
    # Public API Methods
    # ========================================================================

    async def list_converters(
        self, source: Optional[Literal["initializer", "user"]] = None
    ) -> ConverterInstanceListResponse:
        """List all converter instances."""
        items = list(self._instances.values())
        if source is not None:
            items = [c for c in items if c.source == source]
        return ConverterInstanceListResponse(items=items)

    async def get_converter(self, converter_id: str) -> Optional[ConverterInstance]:
        """Get a converter instance by ID."""
        return self._instances.get(converter_id)

    def get_converter_object(self, converter_id: str) -> Optional[Any]:
        """Get the actual converter object."""
        return self._converter_objects.get(converter_id)

    async def create_converter(
        self, request: CreateConverterRequest
    ) -> CreateConverterResponse:
        """Create a new converter instance with optional nested converters."""
        config = {"type": request.type, "params": request.params}
        converter_id, _, created_instances = self._create_converter_recursive(config, "user")

        if request.display_name and converter_id in self._instances:
            self._instances[converter_id].display_name = request.display_name

        outer_instance = self._instances[converter_id]
        return CreateConverterResponse(
            converter_id=converter_id,
            type=request.type,
            display_name=request.display_name,
            params=outer_instance.params,
            created_converters=created_instances if len(created_instances) > 1 else None,
            created_at=outer_instance.created_at,
            source="user",
        )

    async def delete_converter(self, converter_id: str) -> bool:
        """Delete a converter instance."""
        if converter_id in self._instances:
            del self._instances[converter_id]
            self._converter_objects.pop(converter_id, None)
            return True
        return False

    async def preview_conversion(
        self, request: ConverterPreviewRequest
    ) -> ConverterPreviewResponse:
        """Preview conversion through a converter pipeline."""
        converters = self._gather_converters_for_preview(request)
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
        """Get converter objects for a list of IDs."""
        converters = []
        for conv_id in converter_ids:
            conv_obj = self.get_converter_object(conv_id)
            if conv_obj is None:
                raise ValueError(f"Converter instance '{conv_id}' not found")
            converters.append(conv_obj)
        return converters

    def instantiate_inline_converters(self, configs: List[InlineConverterConfig]) -> List[Any]:
        """Instantiate converters from inline configurations."""
        return [
            self._get_converter_class(config.type)(**config.params)
            for config in configs
        ]

    # ========================================================================
    # Private Helper Methods - Class Resolution
    # ========================================================================

    def _get_converter_class(self, converter_type: str) -> type:
        """Get the converter class for a given type."""
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
        """Generate class name patterns to try."""
        pascal = "".join(word.capitalize() for word in type_name.split("_"))
        return [type_name, f"{type_name}Converter", pascal, f"{pascal}Converter"]

    # ========================================================================
    # Private Helper Methods - Recursive Creation
    # ========================================================================

    def _create_converter_recursive(
        self,
        config: Dict[str, Any],
        source: Literal["initializer", "user"],
    ) -> Tuple[str, Any, List[ConverterInstance]]:
        """Recursively create converters, handling nested converter params."""
        converter_type = config["type"]
        params = dict(config.get("params", {}))
        created_instances: List[ConverterInstance] = []

        # Handle nested converter
        params, created_instances = self._resolve_nested_converter(params, source)

        # Create this converter
        converter_obj = self._get_converter_class(converter_type)(**params)
        converter_id = self._store_converter(converter_type, converter_obj, config, created_instances, source)

        return converter_id, converter_obj, created_instances

    def _resolve_nested_converter(
        self,
        params: Dict[str, Any],
        source: Literal["initializer", "user"],
    ) -> Tuple[Dict[str, Any], List[ConverterInstance]]:
        """Resolve nested converter in params if present."""
        created_instances: List[ConverterInstance] = []

        if "converter" in params and isinstance(params["converter"], dict):
            nested_config = params["converter"]
            if "type" in nested_config:
                _, nested_obj, nested_instances = self._create_converter_recursive(nested_config, source)
                created_instances.extend(nested_instances)
                params["converter"] = nested_obj

        return params, created_instances

    def _store_converter(
        self,
        converter_type: str,
        converter_obj: Any,
        config: Dict[str, Any],
        created_instances: List[ConverterInstance],
        source: Literal["initializer", "user"],
    ) -> str:
        """Store converter and return its ID."""
        converter_id = str(uuid.uuid4())
        self._converter_objects[converter_id] = converter_obj

        resolved_params = self._build_resolved_params(config, created_instances)
        instance = ConverterInstance(
            converter_id=converter_id,
            type=converter_type,
            display_name=None,
            params=resolved_params,
            created_at=datetime.now(timezone.utc),
            source=source,
        )
        self._instances[converter_id] = instance
        created_instances.append(instance)

        return converter_id

    def _build_resolved_params(
        self, config: Dict[str, Any], created_instances: List[ConverterInstance]
    ) -> Dict[str, Any]:
        """Build resolved params with nested converter IDs."""
        resolved_params = dict(config.get("params", {}))
        if "converter" in resolved_params and isinstance(resolved_params["converter"], dict):
            nested_id = created_instances[-1].converter_id if created_instances else None
            resolved_params["converter"] = {"converter_id": nested_id}
        return resolved_params

    # ========================================================================
    # Private Helper Methods - Preview
    # ========================================================================

    def _gather_converters_for_preview(
        self, request: ConverterPreviewRequest
    ) -> List[Tuple[Optional[str], str, Any]]:
        """Gather converters to apply from request."""
        converters: List[Tuple[Optional[str], str, Any]] = []

        if request.converter_ids:
            for conv_id in request.converter_ids:
                conv_obj = self.get_converter_object(conv_id)
                if conv_obj is None:
                    raise ValueError(f"Converter instance '{conv_id}' not found")
                instance = self._instances[conv_id]
                converters.append((conv_id, instance.type, conv_obj))

        if request.converters:
            for inline_config in request.converters:
                conv_obj = self._get_converter_class(inline_config.type)(**inline_config.params)
                converters.append((None, inline_config.type, conv_obj))

        return converters

    async def _apply_converters(
        self,
        converters: List[Tuple[Optional[str], str, Any]],
        initial_value: str,
        initial_type: PromptDataType,
    ) -> Tuple[List[PreviewStep], str, PromptDataType]:
        """Apply converters and collect steps."""
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
