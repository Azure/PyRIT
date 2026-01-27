# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter service for managing converter instances.

Handles creation, retrieval, and nested converter support.
"""

import importlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

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
        # In-memory storage for converter instances
        self._instances: Dict[str, ConverterInstance] = {}
        # Actual instantiated converter objects
        self._converter_objects: Dict[str, Any] = {}

    def _get_converter_class(self, converter_type: str) -> type:
        """
        Get the converter class for a given type.

        Args:
            converter_type: Converter type string (e.g., 'base64', 'Base64Converter')

        Returns:
            The converter class
        """
        module = importlib.import_module("pyrit.prompt_converter")

        # Try direct attribute lookup first
        cls = getattr(module, converter_type, None)
        if cls is not None:
            return cls

        # Try common class name patterns
        class_name_patterns = [
            converter_type,
            f"{converter_type}Converter",
            "".join(word.capitalize() for word in converter_type.split("_")),
            "".join(word.capitalize() for word in converter_type.split("_")) + "Converter",
        ]

        for pattern in class_name_patterns:
            cls = getattr(module, pattern, None)
            if cls is not None:
                return cls

        raise ValueError(f"Converter type '{converter_type}' not found in pyrit.prompt_converter")

    def _create_converter_recursive(
        self,
        config: Dict[str, Any],
        source: Literal["initializer", "user"],
    ) -> Tuple[str, Any, List[ConverterInstance]]:
        """
        Recursively create converters, handling nested converter params.

        Args:
            config: Converter configuration with 'type' and 'params'
            source: Source of creation

        Returns:
            Tuple of (converter_id, converter_object, list of all created instances)
        """
        converter_type = config["type"]
        params = dict(config.get("params", {}))
        created_instances: List[ConverterInstance] = []

        # Check for nested converter in params
        if "converter" in params and isinstance(params["converter"], dict):
            nested_config = params["converter"]
            if "type" in nested_config:
                # Recursively create nested converter
                nested_id, nested_obj, nested_instances = self._create_converter_recursive(
                    nested_config, source
                )
                created_instances.extend(nested_instances)
                # Replace inline config with the actual converter object
                params["converter"] = nested_obj

        # Create this converter
        converter_class = self._get_converter_class(converter_type)
        converter_obj = converter_class(**params)

        converter_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Store the converter object
        self._converter_objects[converter_id] = converter_obj

        # Build resolved params (with nested converter IDs instead of objects)
        resolved_params = dict(config.get("params", {}))
        if "converter" in resolved_params and isinstance(resolved_params["converter"], dict):
            # Replace with the nested converter ID
            nested_id = created_instances[-1].converter_id if created_instances else None
            resolved_params["converter"] = {"converter_id": nested_id}

        instance = ConverterInstance(
            converter_id=converter_id,
            type=converter_type,
            display_name=None,
            params=resolved_params,
            created_at=now,
            source=source,
        )
        self._instances[converter_id] = instance
        created_instances.append(instance)

        return converter_id, converter_obj, created_instances

    async def list_converters(
        self,
        source: Optional[Literal["initializer", "user"]] = None,
    ) -> ConverterInstanceListResponse:
        """
        List all converter instances.

        Args:
            source: Optional filter by source

        Returns:
            ConverterInstanceListResponse: List of converter instances
        """
        items = list(self._instances.values())

        if source is not None:
            items = [c for c in items if c.source == source]

        return ConverterInstanceListResponse(items=items)

    async def get_converter(self, converter_id: str) -> Optional[ConverterInstance]:
        """
        Get a converter instance by ID.

        Args:
            converter_id: Converter instance ID

        Returns:
            ConverterInstance or None if not found
        """
        return self._instances.get(converter_id)

    def get_converter_object(self, converter_id: str) -> Optional[Any]:
        """
        Get the actual converter object.

        Args:
            converter_id: Converter instance ID

        Returns:
            The instantiated converter object or None
        """
        return self._converter_objects.get(converter_id)

    async def create_converter(
        self,
        request: CreateConverterRequest,
    ) -> CreateConverterResponse:
        """
        Create a new converter instance.

        Supports nested converters - if params contains a 'converter' key with
        a type/params dict, the nested converter will be created first.

        Args:
            request: Converter creation request

        Returns:
            CreateConverterResponse: Created converter details
        """
        config = {
            "type": request.type,
            "params": request.params,
        }

        converter_id, converter_obj, created_instances = self._create_converter_recursive(
            config, "user"
        )

        # Update display name for the outermost converter
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
        """
        Delete a converter instance.

        Args:
            converter_id: Converter instance ID

        Returns:
            True if deleted, False if not found
        """
        if converter_id in self._instances:
            del self._instances[converter_id]
            self._converter_objects.pop(converter_id, None)
            return True
        return False

    async def preview_conversion(
        self,
        request: ConverterPreviewRequest,
    ) -> ConverterPreviewResponse:
        """
        Preview conversion through a converter pipeline.

        Args:
            request: Preview request with content and converters

        Returns:
            ConverterPreviewResponse: Conversion results with steps
        """
        current_value = request.original_value
        current_type: PromptDataType = request.original_value_data_type
        steps: List[PreviewStep] = []

        # Get converters to apply
        converters_to_apply: List[Tuple[Optional[str], str, Any]] = []

        if request.converter_ids:
            for conv_id in request.converter_ids:
                conv_obj = self.get_converter_object(conv_id)
                if conv_obj is None:
                    raise ValueError(f"Converter instance '{conv_id}' not found")
                instance = self._instances[conv_id]
                converters_to_apply.append((conv_id, instance.type, conv_obj))

        if request.converters:
            for inline_config in request.converters:
                converter_class = self._get_converter_class(inline_config.type)
                conv_obj = converter_class(**inline_config.params)
                converters_to_apply.append((None, inline_config.type, conv_obj))

        # Apply converters in sequence
        for conv_id, conv_type, conv_obj in converters_to_apply:
            input_value = current_value
            input_type = current_type

            result = await conv_obj.convert_async(prompt=current_value)
            current_value = result.output_text
            current_type = result.output_type

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

        return ConverterPreviewResponse(
            original_value=request.original_value,
            original_value_data_type=request.original_value_data_type,
            converted_value=current_value,
            converted_value_data_type=current_type,
            steps=steps,
        )

    def get_converter_objects_for_ids(self, converter_ids: List[str]) -> List[Any]:
        """
        Get converter objects for a list of IDs.

        Args:
            converter_ids: List of converter instance IDs

        Returns:
            List of converter objects

        Raises:
            ValueError: If any converter ID is not found
        """
        converters = []
        for conv_id in converter_ids:
            conv_obj = self.get_converter_object(conv_id)
            if conv_obj is None:
                raise ValueError(f"Converter instance '{conv_id}' not found")
            converters.append(conv_obj)
        return converters

    def instantiate_inline_converters(
        self, configs: List[InlineConverterConfig]
    ) -> List[Any]:
        """
        Instantiate converters from inline configurations.

        Args:
            configs: List of inline converter configs

        Returns:
            List of converter objects
        """
        converters = []
        for config in configs:
            converter_class = self._get_converter_class(config.type)
            conv_obj = converter_class(**config.params)
            converters.append(conv_obj)
        return converters


# Global service instance
_converter_service: Optional[ConverterService] = None


def get_converter_service() -> ConverterService:
    """Get the global converter service instance."""
    global _converter_service
    if _converter_service is None:
        _converter_service = ConverterService()
    return _converter_service
