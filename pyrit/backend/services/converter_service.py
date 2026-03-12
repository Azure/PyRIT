# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converter service for managing converter instances.

Handles creation, retrieval, and preview of converters.
Uses ConverterRegistry as the source of truth for instances.

Converters can be:
- Created via API request (instantiated from request params, then registered)
- Retrieved from registry (pre-registered at startup or created earlier)
"""

import inspect
import pathlib
import uuid
from functools import lru_cache
from typing import Any, Optional, get_args, get_origin, Literal

from pyrit import prompt_converter
from pyrit.backend.mappers.converter_mappers import converter_object_to_instance
from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterInstanceListResponse,
    ConverterParameterMetadata,
    ConverterPreviewRequest,
    ConverterPreviewResponse,
    ConverterTypeListResponse,
    ConverterTypeMetadata,
    ConverterTypePreviewRequest,
    CreateConverterRequest,
    CreateConverterResponse,
    PreviewStep,
)
from pyrit.common.path import CONVERTER_SEED_PROMPT_PATH
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.registry.instance_registries import ConverterRegistry


def _build_converter_class_registry() -> dict[str, type]:
    """
    Build a registry mapping converter class names to their classes.

    Uses the prompt_converter module's __all__ to discover all available converters.

    Returns:
        Dict mapping class name (str) to class (type).
    """
    registry: dict[str, type] = {}
    for name in prompt_converter.__all__:
        cls = getattr(prompt_converter, name, None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, PromptConverter):
            registry[name] = cls
    return registry


# Module-level class registry (built once on import)
_CONVERTER_CLASS_REGISTRY: dict[str, type] = _build_converter_class_registry()


def _dataset_select_options(converter_type: str, parameter_name: str) -> Optional[list[str]]:
    """Expose known string parameters as dropdowns when choices live in the dataset folders."""
    if converter_type == "PersuasionConverter" and parameter_name == "persuasion_technique":
        persuasion_path = pathlib.Path(CONVERTER_SEED_PROMPT_PATH) / "persuasion"
        return sorted(file.stem for file in persuasion_path.glob("*.yaml"))

    return None


def _friendly_name(name: str) -> str:
    """Convert class names into short readable labels."""
    base_name = name.removesuffix("Converter")
    words = []
    current = ""
    for char in base_name:
        if char.isupper() and current and not current[-1].isupper():
            words.append(current)
            current = char
        else:
            current += char
    if current:
        words.append(current)
    return " ".join(words) or name


def _first_sentence(text: Optional[str], *, fallback: str) -> str:
    """Extract a short summary line from a docstring."""
    if not text:
        return fallback
    cleaned = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if not cleaned:
        return fallback
    first_line = cleaned.split(". ")[0].strip()
    return first_line if first_line.endswith(".") else f"{first_line}."


def _stringify_default(value: Any) -> Optional[str]:
    """Convert default values into small readable strings for the UI."""
    if value is inspect.Signature.empty:
        return None
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return repr(value)


def _annotation_to_label(annotation: Any) -> str:
    """Convert type annotations into short readable labels."""
    if annotation is inspect.Signature.empty:
        return "Any"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return "one of: " + ", ".join(str(arg) for arg in args)

    if origin in {list, tuple} and args:
        return f"list of {_annotation_to_label(args[0]).lower()}"

    if origin is dict:
        return "object"

    if origin is not None and type(None) in args:
        inner = [arg for arg in args if arg is not type(None)]
        if len(inner) == 1:
            return f"optional {_annotation_to_label(inner[0]).lower()}"
        return "optional value"

    if annotation in {str, int, float, bool}:
        return annotation.__name__

    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def _annotation_to_input_kind(annotation: Any) -> tuple[str, Optional[list[str]]]:
    """Suggest a simple UI control for a parameter annotation."""
    if annotation is inspect.Signature.empty:
        return "text", None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return "select", [str(arg) for arg in args]

    if origin is not None and type(None) in args:
        inner = [arg for arg in args if arg is not type(None)]
        if len(inner) == 1:
            return _annotation_to_input_kind(inner[0])

    if annotation is bool:
        return "boolean", None
    if annotation in {int, float}:
        return "number", None
    if annotation is str:
        return "text", None

    if origin in {list, tuple}:
        inner = args[0] if args else str
        if inner in {str, int, float}:
            return "list", None
        return "unsupported", None

    return "unsupported", None


def _parameter_metadata(*, converter_type: str, parameter: inspect.Parameter) -> ConverterParameterMetadata:
    """Build UI metadata for a constructor parameter."""
    input_kind, options = _annotation_to_input_kind(parameter.annotation)
    dataset_options = _dataset_select_options(converter_type, parameter.name)
    if dataset_options:
        input_kind = "select"
        options = dataset_options

    return ConverterParameterMetadata(
        name=parameter.name,
        display_name=parameter.name.replace("_", " ").title(),
        type_label=_annotation_to_label(parameter.annotation),
        required=parameter.default is inspect.Signature.empty,
        default_value=_stringify_default(parameter.default),
        input_kind=input_kind,
        options=options,
    )


class ConverterService:
    """
    Service for managing converter instances.

    Uses ConverterRegistry as the sole source of truth.
    API metadata is derived from the converter objects.
    """

    def __init__(self) -> None:
        """Initialize the converter service."""
        self._registry = ConverterRegistry.get_registry_singleton()

    def _build_instance_from_object(self, *, converter_id: str, converter_obj: Any) -> ConverterInstance:
        """
        Build a ConverterInstance from a registry object.

        Uses the converter's identifier to extract all relevant metadata.

        Returns:
            ConverterInstance with metadata derived from the object's identifier.
        """
        return converter_object_to_instance(converter_id, converter_obj)

    # ========================================================================
    # Public API Methods
    # ========================================================================

    async def list_converters_async(self) -> ConverterInstanceListResponse:
        """
        List all converter instances.

        Returns:
            ConverterInstanceListResponse containing all registered converters.
        """
        items = [
            self._build_instance_from_object(converter_id=name, converter_obj=obj)
            for name, obj in self._registry.get_all_instances().items()
        ]
        return ConverterInstanceListResponse(items=items)

    async def list_converter_types_async(self) -> ConverterTypeListResponse:
        """
        List all available converter classes with lightweight UI metadata.

        Returns:
            ConverterTypeListResponse containing all available converter types.
        """
        items: list[ConverterTypeMetadata] = []

        for converter_type, converter_class in sorted(_CONVERTER_CLASS_REGISTRY.items()):
            if inspect.isabstract(converter_class):
                continue

            signature = inspect.signature(converter_class.__init__)
            parameters = [
                _parameter_metadata(converter_type=converter_type, parameter=parameter)
                for name, parameter in signature.parameters.items()
                if name != "self" and parameter.kind not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
            ]
            required_unsupported = [param.display_name for param in parameters if param.required and param.input_kind == "unsupported"]

            items.append(
                ConverterTypeMetadata(
                    converter_type=converter_type,
                    display_name=_friendly_name(converter_type),
                    description=_first_sentence(
                        inspect.getdoc(converter_class),
                        fallback="Transforms the prompt before it is sent to the target.",
                    ),
                    supported_input_types=[str(value) for value in getattr(converter_class, "SUPPORTED_INPUT_TYPES", ())],
                    supported_output_types=[str(value) for value in getattr(converter_class, "SUPPORTED_OUTPUT_TYPES", ())],
                    parameters=parameters,
                    preview_supported=not required_unsupported,
                    preview_unavailable_reason=(
                        "Needs extra setup for: " + ", ".join(required_unsupported)
                        if required_unsupported
                        else None
                    ),
                )
            )

        return ConverterTypeListResponse(items=items)

    async def get_converter_async(self, *, converter_id: str) -> Optional[ConverterInstance]:
        """
        Get a converter instance by ID.

        Returns:
            ConverterInstance if found, None otherwise.
        """
        obj = self._registry.get_instance_by_name(converter_id)
        if obj is None:
            return None
        return self._build_instance_from_object(converter_id=converter_id, converter_obj=obj)

    def get_converter_object(self, *, converter_id: str) -> Optional[Any]:
        """
        Get the actual converter object.

        Returns:
            The PromptConverter object if found, None otherwise.
        """
        return self._registry.get_instance_by_name(converter_id)

    async def create_converter_async(self, *, request: CreateConverterRequest) -> CreateConverterResponse:
        """
        Create a new converter instance from API request.

        Instantiates the converter with the given type and params,
        then registers it in the registry.

        Args:
            request: The create converter request with type and params.

        Returns:
            CreateConverterResponse with the new converter's details.

        Raises:
            ValueError: If the converter type is not found.
        """
        converter_id = str(uuid.uuid4())

        # Resolve any converter references in params and instantiate
        params = self._resolve_converter_params(params=request.params)
        converter_class = self._get_converter_class(converter_type=request.type)
        converter_obj = converter_class(**params)
        self._registry.register_instance(converter_obj, name=converter_id)

        return CreateConverterResponse(
            converter_id=converter_id,
            converter_type=request.type,
            display_name=request.display_name,
        )

    async def preview_conversion_async(self, *, request: ConverterPreviewRequest) -> ConverterPreviewResponse:
        """
        Preview conversion through a converter pipeline.

        Returns:
            ConverterPreviewResponse with step-by-step conversion results.
        """
        converters = self._gather_converters(converter_ids=request.converter_ids)
        steps, final_value, final_type = await self._apply_converters(
            converters=converters, initial_value=request.original_value, initial_type=request.original_value_data_type
        )

        return ConverterPreviewResponse(
            original_value=request.original_value,
            original_value_data_type=request.original_value_data_type,
            converted_value=final_value,
            converted_value_data_type=final_type,
            steps=steps,
        )

    async def preview_converter_type_async(self, *, request: ConverterTypePreviewRequest) -> ConverterPreviewResponse:
        """
        Preview a converter directly from its type and params without registering it.

        Returns:
            ConverterPreviewResponse containing a single-step preview.
        """
        params = self._resolve_converter_params(params=request.params)
        converter_class = self._get_converter_class(converter_type=request.type)
        converter_obj = converter_class(**params)
        result = await converter_obj.convert_async(
            prompt=request.original_value,
            input_type=request.original_value_data_type,
        )

        step = PreviewStep(
            converter_id=request.type,
            converter_type=request.type,
            input_value=request.original_value,
            input_data_type=request.original_value_data_type,
            output_value=result.output_text,
            output_data_type=result.output_type,
        )

        return ConverterPreviewResponse(
            original_value=request.original_value,
            original_value_data_type=request.original_value_data_type,
            converted_value=result.output_text,
            converted_value_data_type=result.output_type,
            steps=[step],
        )

    def get_converter_objects_for_ids(self, *, converter_ids: list[str]) -> list[Any]:
        """
        Get converter objects for a list of IDs.

        Returns:
            List of converter objects in the same order as the input IDs.
        """
        converters = []
        for conv_id in converter_ids:
            conv_obj = self.get_converter_object(converter_id=conv_id)
            if conv_obj is None:
                raise ValueError(f"Converter instance '{conv_id}' not found")
            converters.append(conv_obj)
        return converters

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _get_converter_class(self, *, converter_type: str) -> type:
        """
        Get the converter class for a given type name.

        Looks up the class in the module-level converter class registry.

        Args:
            converter_type: The exact class name of the converter (e.g., 'Base64Converter').

        Returns:
            The converter class.

        Raises:
            ValueError: If the converter type is not found.
        """
        cls = _CONVERTER_CLASS_REGISTRY.get(converter_type)
        if cls is None:
            raise ValueError(
                f"Converter type '{converter_type}' not found. "
                f"Available types: {sorted(_CONVERTER_CLASS_REGISTRY.keys())}"
            )
        return cls

    def _resolve_converter_params(self, *, params: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve converter references in params.

        If params contains a 'converter' key with a converter_id reference,
        resolve it to the actual converter object from the registry.

        Returns:
            Params dict with converter_id references replaced by actual objects.
        """
        resolved = dict(params)
        if "converter" in resolved and isinstance(resolved["converter"], dict):
            ref = resolved["converter"]
            if "converter_id" in ref:
                conv_obj = self.get_converter_object(converter_id=ref["converter_id"])
                if conv_obj is None:
                    raise ValueError(f"Referenced converter '{ref['converter_id']}' not found")
                resolved["converter"] = conv_obj
        return resolved

    def _gather_converters(self, *, converter_ids: list[str]) -> list[tuple[str, str, Any]]:
        """
        Gather converters to apply from IDs.

        Returns:
            List of tuples (converter_id, converter_type, converter_obj).
        """
        converters: list[tuple[str, str, Any]] = []
        for conv_id in converter_ids:
            conv_obj = self.get_converter_object(converter_id=conv_id)
            if conv_obj is None:
                raise ValueError(f"Converter instance '{conv_id}' not found")
            conv_type = conv_obj.__class__.__name__
            converters.append((conv_id, conv_type, conv_obj))
        return converters

    async def _apply_converters(
        self,
        *,
        converters: list[tuple[str, str, Any]],
        initial_value: str,
        initial_type: PromptDataType,
    ) -> tuple[list[PreviewStep], str, PromptDataType]:
        """
        Apply converters and collect steps.

        Returns:
            Tuple of (steps, final_value, final_type).
        """
        current_value = initial_value
        current_type = initial_type
        steps: list[PreviewStep] = []

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


@lru_cache(maxsize=1)
def get_converter_service() -> ConverterService:
    """
    Get the global converter service instance.

    Returns:
        The singleton ConverterService instance.
    """
    return ConverterService()
