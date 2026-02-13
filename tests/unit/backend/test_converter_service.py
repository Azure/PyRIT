# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend converter service.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit import prompt_converter
from pyrit.backend.models.converters import (
    ConverterPreviewRequest,
    CreateConverterRequest,
)
from pyrit.backend.services.converter_service import ConverterService, get_converter_service
from pyrit.prompt_converter import (
    Base64Converter,
    CaesarConverter,
    RepeatTokenConverter,
    SuffixAppendConverter,
)
from pyrit.prompt_converter.prompt_converter import get_converter_modalities
from pyrit.registry.instance_registries import ConverterRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the ConverterRegistry singleton before each test."""
    ConverterRegistry.reset_instance()
    yield
    ConverterRegistry.reset_instance()


class TestListConverters:
    """Tests for ConverterService.list_converters method."""

    @pytest.mark.asyncio
    async def test_list_converters_returns_empty_when_no_converters(self) -> None:
        """Test that list_converters returns empty list when no converters exist."""
        service = ConverterService()

        result = await service.list_converters_async()

        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_converters_returns_converters_from_registry(self) -> None:
        """Test that list_converters returns converters from registry with full params."""
        service = ConverterService()

        mock_converter = MagicMock()
        mock_converter.__class__.__name__ = "MockConverter"
        mock_identifier = MagicMock()
        mock_identifier.class_name = "MockConverter"
        mock_identifier.supported_input_types = ("text",)
        mock_identifier.supported_output_types = ("text",)
        mock_identifier.converter_specific_params = {"param1": "value1", "param2": 42}
        mock_identifier.sub_identifier = None
        mock_converter.get_identifier.return_value = mock_identifier
        service._registry.register_instance(mock_converter, name="conv-1")

        result = await service.list_converters_async()

        assert len(result.items) == 1
        assert result.items[0].converter_id == "conv-1"
        assert result.items[0].converter_type == "MockConverter"
        assert result.items[0].supported_input_types == ["text"]
        assert result.items[0].supported_output_types == ["text"]
        assert result.items[0].converter_specific_params == {"param1": "value1", "param2": 42}


class TestGetConverter:
    """Tests for ConverterService.get_converter method."""

    @pytest.mark.asyncio
    async def test_get_converter_returns_none_for_nonexistent(self) -> None:
        """Test that get_converter returns None for non-existent converter."""
        service = ConverterService()

        result = await service.get_converter_async(converter_id="nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_converter_returns_converter_from_registry(self) -> None:
        """Test that get_converter returns converter built from registry object."""
        service = ConverterService()

        mock_converter = MagicMock()
        mock_converter.__class__.__name__ = "MockConverter"
        mock_identifier = MagicMock()
        mock_identifier.class_name = "MockConverter"
        mock_identifier.supported_input_types = ("text",)
        mock_identifier.supported_output_types = ("text",)
        mock_identifier.converter_specific_params = {"param1": "value1"}
        mock_identifier.sub_identifier = None
        mock_converter.get_identifier.return_value = mock_identifier
        service._registry.register_instance(mock_converter, name="conv-1")

        result = await service.get_converter_async(converter_id="conv-1")

        assert result is not None
        assert result.converter_id == "conv-1"
        assert result.converter_type == "MockConverter"


class TestGetConverterObject:
    """Tests for ConverterService.get_converter_object method."""

    def test_get_converter_object_returns_none_for_nonexistent(self) -> None:
        """Test that get_converter_object returns None for non-existent converter."""
        service = ConverterService()

        result = service.get_converter_object(converter_id="nonexistent-id")

        assert result is None

    def test_get_converter_object_returns_object_from_registry(self) -> None:
        """Test that get_converter_object returns the actual converter object."""
        service = ConverterService()
        mock_converter = MagicMock()
        service._registry.register_instance(mock_converter, name="conv-1")

        result = service.get_converter_object(converter_id="conv-1")

        assert result is mock_converter


class TestCreateConverter:
    """Tests for ConverterService.create_converter method."""

    @pytest.mark.asyncio
    async def test_create_converter_raises_for_invalid_type(self) -> None:
        """Test that create_converter raises for invalid converter type."""
        service = ConverterService()

        request = CreateConverterRequest(
            type="NonExistentConverter",
            params={},
        )

        with pytest.raises(ValueError, match="not found"):
            await service.create_converter_async(request=request)

    @pytest.mark.asyncio
    async def test_create_converter_success(self) -> None:
        """Test successful converter creation."""
        service = ConverterService()

        request = CreateConverterRequest(
            type="Base64Converter",
            display_name="My Base64",
            params={},
        )

        result = await service.create_converter_async(request=request)

        assert result.converter_id is not None
        assert result.converter_type == "Base64Converter"
        assert result.display_name == "My Base64"

    @pytest.mark.asyncio
    async def test_create_converter_registers_in_registry(self) -> None:
        """Test that create_converter registers object in registry."""
        service = ConverterService()

        request = CreateConverterRequest(
            type="Base64Converter",
            params={},
        )

        result = await service.create_converter_async(request=request)

        # Object should be retrievable from registry
        converter_obj = service.get_converter_object(converter_id=result.converter_id)
        assert converter_obj is not None


class TestResolveConverterParams:
    """Tests for ConverterService._resolve_converter_params method."""

    def test_resolve_converter_params_returns_params_unchanged_when_no_converter_ref(self) -> None:
        """Test that params without converter reference are returned unchanged."""
        service = ConverterService()
        params = {"key": "value", "number": 42}

        result = service._resolve_converter_params(params=params)

        assert result == params

    def test_resolve_converter_params_resolves_converter_id_reference(self) -> None:
        """Test that converter_id reference is resolved to actual object."""
        service = ConverterService()

        # Register a mock converter
        mock_converter = MagicMock()
        service._registry.register_instance(mock_converter, name="inner-conv")

        params = {"converter": {"converter_id": "inner-conv"}}

        result = service._resolve_converter_params(params=params)

        assert result["converter"] is mock_converter

    def test_resolve_converter_params_raises_for_nonexistent_reference(self) -> None:
        """Test that referencing a non-existent converter raises ValueError."""
        service = ConverterService()

        params = {"converter": {"converter_id": "nonexistent"}}

        with pytest.raises(ValueError, match="not found"):
            service._resolve_converter_params(params=params)

    def test_resolve_converter_params_ignores_non_dict_converter(self) -> None:
        """Test that non-dict converter values are not modified."""
        service = ConverterService()
        params = {"converter": "some_string_value"}

        result = service._resolve_converter_params(params=params)

        assert result == params


class TestPreviewConversion:
    """Tests for ConverterService.preview_conversion method."""

    @pytest.mark.asyncio
    async def test_preview_conversion_raises_for_nonexistent_converter(self) -> None:
        """Test that preview raises ValueError for non-existent converter ID."""
        service = ConverterService()

        request = ConverterPreviewRequest(
            original_value="test",
            original_value_data_type="text",
            converter_ids=["nonexistent"],
        )

        with pytest.raises(ValueError, match="not found"):
            await service.preview_conversion_async(request=request)

    @pytest.mark.asyncio
    async def test_preview_conversion_with_converter_ids(self) -> None:
        """Test preview with converter IDs."""
        service = ConverterService()

        mock_converter = MagicMock()
        mock_converter.__class__.__name__ = "MockConverter"
        mock_result = MagicMock()
        mock_result.output_text = "encoded_value"
        mock_result.output_type = "text"
        mock_converter.convert_async = AsyncMock(return_value=mock_result)
        service._registry.register_instance(mock_converter, name="conv-1")

        request = ConverterPreviewRequest(
            original_value="test",
            original_value_data_type="text",
            converter_ids=["conv-1"],
        )

        result = await service.preview_conversion_async(request=request)

        assert result.original_value == "test"
        assert result.converted_value == "encoded_value"
        assert len(result.steps) == 1
        assert result.steps[0].converter_id == "conv-1"

    @pytest.mark.asyncio
    async def test_preview_conversion_chains_multiple_converters(self) -> None:
        """Test that preview chains multiple converters."""
        service = ConverterService()

        mock_converter1 = MagicMock()
        mock_converter1.__class__.__name__ = "MockConverter1"
        mock_result1 = MagicMock()
        mock_result1.output_text = "step1_output"
        mock_result1.output_type = "text"
        mock_converter1.convert_async = AsyncMock(return_value=mock_result1)

        mock_converter2 = MagicMock()
        mock_converter2.__class__.__name__ = "MockConverter2"
        mock_result2 = MagicMock()
        mock_result2.output_text = "step2_output"
        mock_result2.output_type = "text"
        mock_converter2.convert_async = AsyncMock(return_value=mock_result2)

        service._registry.register_instance(mock_converter1, name="conv-1")
        service._registry.register_instance(mock_converter2, name="conv-2")

        request = ConverterPreviewRequest(
            original_value="input",
            original_value_data_type="text",
            converter_ids=["conv-1", "conv-2"],
        )

        result = await service.preview_conversion_async(request=request)

        assert result.converted_value == "step2_output"
        assert len(result.steps) == 2
        mock_converter2.convert_async.assert_called_with(prompt="step1_output")


class TestGetConverterObjectsForIds:
    """Tests for ConverterService.get_converter_objects_for_ids method."""

    def test_get_converter_objects_for_ids_raises_for_nonexistent(self) -> None:
        """Test that method raises ValueError for non-existent ID."""
        service = ConverterService()

        with pytest.raises(ValueError, match="not found"):
            service.get_converter_objects_for_ids(converter_ids=["nonexistent"])

    def test_get_converter_objects_for_ids_returns_objects(self) -> None:
        """Test that method returns converter objects in order."""
        service = ConverterService()

        mock1 = MagicMock()
        mock2 = MagicMock()
        service._registry.register_instance(mock1, name="conv-1")
        service._registry.register_instance(mock2, name="conv-2")

        result = service.get_converter_objects_for_ids(converter_ids=["conv-1", "conv-2"])

        assert result == [mock1, mock2]


class TestConverterServiceSingleton:
    """Tests for get_converter_service singleton function."""

    def test_get_converter_service_returns_converter_service(self) -> None:
        """Test that get_converter_service returns a ConverterService instance."""
        get_converter_service.cache_clear()

        service = get_converter_service()
        assert isinstance(service, ConverterService)

    def test_get_converter_service_returns_same_instance(self) -> None:
        """Test that get_converter_service returns the same instance."""
        get_converter_service.cache_clear()

        service1 = get_converter_service()
        service2 = get_converter_service()
        assert service1 is service2


# ============================================================================
# Real Converter Integration Tests
# ============================================================================


def _get_all_converter_names() -> list[str]:
    """
    Dynamically collect all converter class names from the codebase.

    Uses get_converter_modalities() which reads from prompt_converter.__all__
    and filters to only actual PromptConverter subclasses.
    """
    return [name for name, _, _ in get_converter_modalities()]


def _try_instantiate_converter(converter_name: str):
    """
    Try to instantiate a converter with no arguments.

    Returns:
        Tuple of (converter_instance, error_message).
        If successful, error_message is None.
        If failed, converter_instance is None and error_message explains why.
    """
    converter_cls = getattr(prompt_converter, converter_name, None)
    if converter_cls is None:
        return None, f"Converter {converter_name} not found in prompt_converter module"

    try:
        instance = converter_cls()
        return instance, None
    except Exception as e:
        return None, f"Could not instantiate {converter_name} with no args: {e}"


# Get all converter names dynamically
ALL_CONVERTERS = _get_all_converter_names()


class TestBuildInstanceFromObjectWithRealConverters:
    """
    Integration tests that verify _build_instance_from_object works with real converters.

    These tests ensure the identifier extraction works correctly across all converter types.
    Uses dynamic discovery to test ALL converters in the codebase.
    """

    @pytest.mark.parametrize("converter_name", ALL_CONVERTERS)
    def test_build_instance_from_converter(self, converter_name: str) -> None:
        """
        Test that _build_instance_from_object works with each converter.

        For converters that can be instantiated with no arguments, verifies:
        - converter_id is set correctly
        - converter_type matches the class name
        - supported_input_types and supported_output_types are lists

        For converters requiring arguments, the test is skipped (since we can't
        know the required parameters without external configuration).
        """
        # Try to instantiate the converter
        converter_instance, error = _try_instantiate_converter(converter_name)

        if error:
            pytest.skip(error)

        # Build the instance using the service method
        service = ConverterService()
        result = service._build_instance_from_object(converter_id="test-id", converter_obj=converter_instance)

        # Verify the result
        assert result.converter_id == "test-id"
        assert result.converter_type == converter_name
        assert isinstance(result.supported_input_types, list)
        assert isinstance(result.supported_output_types, list)


class TestConverterParamsExtraction:
    """
    Tests that verify converter_specific_params are correctly extracted.

    Uses converters with known parameters to verify the params are properly
    captured from the identifier.
    """

    def test_caesar_converter_params(self) -> None:
        """Test that CaesarConverter params are extracted correctly."""
        converter = CaesarConverter(caesar_offset=13)
        service = ConverterService()
        result = service._build_instance_from_object(converter_id="test-id", converter_obj=converter)

        assert result.converter_type == "CaesarConverter"
        assert result.converter_specific_params is not None
        assert result.converter_specific_params.get("caesar_offset") == 13

    def test_suffix_append_converter_params(self) -> None:
        """Test that SuffixAppendConverter params are extracted correctly."""
        converter = SuffixAppendConverter(suffix="test suffix")
        service = ConverterService()
        result = service._build_instance_from_object(converter_id="test-id", converter_obj=converter)

        assert result.converter_type == "SuffixAppendConverter"
        assert result.converter_specific_params is not None
        assert result.converter_specific_params.get("suffix") == "test suffix"

    def test_repeat_token_converter_params(self) -> None:
        """Test that RepeatTokenConverter params are extracted correctly."""
        converter = RepeatTokenConverter(token_to_repeat="x", times_to_repeat=5)
        service = ConverterService()
        result = service._build_instance_from_object(converter_id="test-id", converter_obj=converter)

        assert result.converter_type == "RepeatTokenConverter"
        assert result.converter_specific_params is not None
        assert result.converter_specific_params.get("token_to_repeat") == "x"
        assert result.converter_specific_params.get("times_to_repeat") == 5

    def test_base64_converter_default_params(self) -> None:
        """Test that Base64Converter default params are captured."""
        converter = Base64Converter()
        service = ConverterService()
        result = service._build_instance_from_object(converter_id="test-id", converter_obj=converter)

        assert result.converter_type == "Base64Converter"
        # Verify type info is populated from identifier
        assert isinstance(result.supported_input_types, list)
        assert isinstance(result.supported_output_types, list)
