# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Registry service for API access to registered components.

Wraps component registries with filtering and metadata extraction.
Uses class introspection when registries are not available.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Type

from pyrit.backend.models.common import filter_sensitive_fields
from pyrit.backend.models.registry import (
    ConverterMetadataResponse,
    InitializerMetadataResponse,
    ScenarioMetadataResponse,
    ScorerMetadataResponse,
    TargetMetadataResponse,
)
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.registry import InitializerRegistry, ScenarioRegistry, ScorerRegistry

logger = logging.getLogger(__name__)


def _extract_params_schema(cls: Type[Any]) -> Dict[str, Any]:
    """
    Extract parameter schema from a class constructor.

    Returns:
        Dict[str, Any]: Dict with 'required' and 'optional' fields.
    """
    required = []
    optional = []

    try:
        sig = inspect.signature(cls.__init__)

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "args", "kwargs"):
                continue

            if param.default == inspect.Parameter.empty:
                required.append(name)
            else:
                optional.append(name)
    except (ValueError, TypeError):
        pass

    return {"required": required, "optional": optional}


def _get_all_subclasses(cls: type) -> List[type]:
    """
    Recursively get all non-abstract subclasses of a class.

    Returns:
        List[type]: List of non-abstract subclass types.
    """
    subclasses = []
    for subclass in cls.__subclasses__():
        # Skip abstract classes
        if hasattr(subclass, "__abstractmethods__") and subclass.__abstractmethods__:
            subclasses.extend(_get_all_subclasses(subclass))
        else:
            subclasses.append(subclass)
            subclasses.extend(_get_all_subclasses(subclass))
    return subclasses


class RegistryService:
    """Service for querying component registries."""

    def get_targets(
        self,
        *,
        is_chat_target: Optional[bool] = None,
    ) -> List[TargetMetadataResponse]:
        """
        Get available target types via introspection.

        Note: TargetRegistry may not exist yet (PR #1320).
        Falls back to class introspection.

        Args:
            is_chat_target: Filter by chat target support.

        Returns:
            List of target metadata.
        """
        # Get all concrete target subclasses via introspection
        target_classes = _get_all_subclasses(PromptTarget)

        results = []
        for target_class in target_classes:
            # Determine if chat target
            is_chat = issubclass(target_class, PromptChatTarget)

            if is_chat_target is not None and is_chat != is_chat_target:
                continue

            # Check JSON response support
            supports_json = False
            if is_chat:
                try:
                    supports_json = hasattr(target_class, "is_json_response_supported")
                except Exception:
                    pass

            # Get supported data types from class attribute if available
            supported_types = getattr(target_class, "SUPPORTED_DATA_TYPES", ["text"])

            results.append(
                TargetMetadataResponse(
                    name=target_class.__name__,
                    class_name=target_class.__name__,
                    description=(target_class.__doc__ or "").split("\n")[0].strip(),
                    is_chat_target=is_chat,
                    supports_json_response=supports_json,
                    supported_data_types=list(supported_types),
                    params_schema=_extract_params_schema(target_class),
                )
            )

        return results

    def get_scenarios(self) -> List[ScenarioMetadataResponse]:
        """
        Get all available scenarios from the registry.

        Returns:
            List of scenario metadata.
        """
        try:
            registry = ScenarioRegistry.get_registry_singleton()
            metadata_list = registry.list_metadata()

            results = []
            for m in metadata_list:
                results.append(
                    ScenarioMetadataResponse(
                        name=m.name,
                        class_name=m.class_name,
                        description=m.class_description or "",
                        default_strategy=m.default_strategy,
                        all_strategies=list(m.all_strategies),
                        aggregate_strategies=list(m.aggregate_strategies),
                        default_datasets=list(m.default_datasets),
                        max_dataset_size=m.max_dataset_size,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"Failed to get scenarios from registry: {e}")
            return []

    def get_scorers(
        self,
        *,
        scorer_type: Optional[str] = None,
    ) -> List[ScorerMetadataResponse]:
        """
        Get registered scorer instances.

        Args:
            scorer_type: Filter by scorer type ('true_false' or 'float_scale').

        Returns:
            List of scorer metadata.
        """
        try:
            registry = ScorerRegistry.get_registry_singleton()

            # Build filter if scorer_type specified
            include_filters: dict[str, object] | None = None
            if scorer_type:
                include_filters = {"scorer_type": scorer_type}

            metadata_list = registry.list_metadata(include_filters=include_filters)

            results = []
            for m in metadata_list:
                # Get scorer identifier and filter sensitive fields
                scorer_id = m.scorer_identifier.to_compact_dict() if m.scorer_identifier else {}
                filtered_id = filter_sensitive_fields(scorer_id)

                results.append(
                    ScorerMetadataResponse(
                        name=m.name,
                        class_name=m.class_name,
                        description=m.class_description or "",
                        scorer_type=m.scorer_type,
                        scorer_identifier=filtered_id,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"Failed to get scorers from registry: {e}")
            return []

    def get_initializers(self) -> List[InitializerMetadataResponse]:
        """
        Get all available initializers from the registry.

        Returns:
            List of initializer metadata.
        """
        try:
            registry = InitializerRegistry.get_registry_singleton()
            metadata_list = registry.list_metadata()

            results = []
            for m in metadata_list:
                results.append(
                    InitializerMetadataResponse(
                        name=m.name,
                        class_name=m.class_name,
                        description=m.class_description or "",
                        required_env_vars=list(m.required_env_vars) if m.required_env_vars else [],
                        execution_order=getattr(m, "execution_order", 0),
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"Failed to get initializers from registry: {e}")
            return []

    def get_converters(
        self,
        *,
        is_llm_based: Optional[bool] = None,
        is_deterministic: Optional[bool] = None,
    ) -> List[ConverterMetadataResponse]:
        """
        Get available converters via introspection.

        Note: ConverterRegistry may not exist yet.
        Falls back to class introspection.

        Args:
            is_llm_based: Filter by LLM-based converters.
            is_deterministic: Filter by deterministic converters.

        Returns:
            List of converter metadata.
        """
        # Get all converter subclasses using the shared helper
        converter_classes = _get_all_subclasses(PromptConverter)

        results = []
        for converter_class in converter_classes:
            # Get supported types from class attributes
            input_types = getattr(converter_class, "SUPPORTED_INPUT_TYPES", ["text"])
            output_types = getattr(converter_class, "SUPPORTED_OUTPUT_TYPES", ["text"])

            # Determine if LLM-based (has converter_target parameter)
            converter_is_llm_based = False
            try:
                sig = inspect.signature(converter_class)
                converter_is_llm_based = "converter_target" in sig.parameters
            except Exception:
                pass

            if is_llm_based is not None and converter_is_llm_based != is_llm_based:
                continue

            # Assume deterministic if not LLM-based
            converter_is_deterministic = not converter_is_llm_based

            if is_deterministic is not None and converter_is_deterministic != is_deterministic:
                continue

            results.append(
                ConverterMetadataResponse(
                    name=converter_class.__name__,
                    class_name=converter_class.__name__,
                    description=(converter_class.__doc__ or "").split("\n")[0].strip(),
                    supported_input_types=list(input_types),
                    supported_output_types=list(output_types),
                    is_llm_based=converter_is_llm_based,
                    is_deterministic=converter_is_deterministic,
                    params_schema=_extract_params_schema(converter_class),
                )
            )

        return results


# Singleton instance
_registry_service: Optional[RegistryService] = None


def get_registry_service() -> RegistryService:
    """
    Get the registry service singleton.

    Returns:
        RegistryService: The registry service instance.
    """
    global _registry_service
    if _registry_service is None:
        _registry_service = RegistryService()
    return _registry_service
