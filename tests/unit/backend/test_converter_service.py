# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend converter service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.models.converters import (
    ConverterPreviewRequest,
    CreateConverterRequest,
)
from pyrit.backend.services.converter_service import ConverterService
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

        result = await service.list_converters()

        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_converters_returns_converters_from_registry(self) -> None:
        """Test that list_converters returns converters from registry."""
        service = ConverterService()

        mock_converter = MagicMock()
        mock_converter.__class__.__name__ = "MockConverter"
        service._registry.register_instance(mock_converter, name="conv-1")

        result = await service.list_converters()

        assert len(result.items) == 1
        assert result.items[0].converter_id == "conv-1"
        assert result.items[0].type == "MockConverter"


class TestGetConverter:
    """Tests for ConverterService.get_converter method."""

    @pytest.mark.asyncio
    async def test_get_converter_returns_none_for_nonexistent(self) -> None:
        """Test that get_converter returns None for non-existent converter."""
        service = ConverterService()

        result = await service.get_converter("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_converter_returns_converter_from_registry(self) -> None:
        """Test that get_converter returns converter built from registry object."""
        service = ConverterService()

        mock_converter = MagicMock()
        mock_converter.__class__.__name__ = "MockConverter"
        service._registry.register_instance(mock_converter, name="conv-1")

        result = await service.get_converter("conv-1")

        assert result is not None
        assert result.converter_id == "conv-1"
        assert result.type == "MockConverter"


class TestGetConverterObject:
    """Tests for ConverterService.get_converter_object method."""

    def test_get_converter_object_returns_none_for_nonexistent(self) -> None:
        """Test that get_converter_object returns None for non-existent converter."""
        service = ConverterService()

        result = service.get_converter_object("nonexistent-id")

        assert result is None

    def test_get_converter_object_returns_object_from_registry(self) -> None:
        """Test that get_converter_object returns the actual converter object."""
        service = ConverterService()
        mock_converter = MagicMock()
        service._registry.register_instance(mock_converter, name="conv-1")

        result = service.get_converter_object("conv-1")

        assert result is mock_converter


class TestGetConverterClass:
    """Tests for ConverterService._get_converter_class method."""

    def test_get_converter_class_raises_for_invalid_type(self) -> None:
        """Test that _get_converter_class raises ValueError for invalid type."""
        service = ConverterService()

        with pytest.raises(ValueError, match="not found"):
            service._get_converter_class("NonExistentConverter")

    def test_get_converter_class_finds_base64_converter(self) -> None:
        """Test that _get_converter_class finds Base64Converter."""
        service = ConverterService()

        result = service._get_converter_class("Base64Converter")

        assert result is not None
        assert "Base64" in result.__name__

    def test_get_converter_class_handles_snake_case(self) -> None:
        """Test that _get_converter_class handles snake_case names."""
        service = ConverterService()

        result = service._get_converter_class("base64")

        assert result is not None


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
            await service.create_converter(request)

    @pytest.mark.asyncio
    async def test_create_converter_success(self) -> None:
        """Test successful converter creation."""
        service = ConverterService()

        request = CreateConverterRequest(
            type="Base64Converter",
            display_name="My Base64",
            params={},
        )

        result = await service.create_converter(request)

        assert result.converter_id is not None
        assert result.type == "Base64Converter"
        assert result.display_name == "My Base64"

    @pytest.mark.asyncio
    async def test_create_converter_registers_in_registry(self) -> None:
        """Test that create_converter registers object in registry."""
        service = ConverterService()

        request = CreateConverterRequest(
            type="Base64Converter",
            params={},
        )

        result = await service.create_converter(request)

        # Object should be retrievable from registry
        converter_obj = service.get_converter_object(result.converter_id)
        assert converter_obj is not None


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
            await service.preview_conversion(request)

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

        result = await service.preview_conversion(request)

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

        result = await service.preview_conversion(request)

        assert result.converted_value == "step2_output"
        assert len(result.steps) == 2
        mock_converter2.convert_async.assert_called_with(prompt="step1_output")


class TestGetConverterObjectsForIds:
    """Tests for ConverterService.get_converter_objects_for_ids method."""

    def test_get_converter_objects_for_ids_raises_for_nonexistent(self) -> None:
        """Test that method raises ValueError for non-existent ID."""
        service = ConverterService()

        with pytest.raises(ValueError, match="not found"):
            service.get_converter_objects_for_ids(["nonexistent"])

    def test_get_converter_objects_for_ids_returns_objects(self) -> None:
        """Test that method returns converter objects in order."""
        service = ConverterService()

        mock1 = MagicMock()
        mock2 = MagicMock()
        service._registry.register_instance(mock1, name="conv-1")
        service._registry.register_instance(mock2, name="conv-2")

        result = service.get_converter_objects_for_ids(["conv-1", "conv-2"])

        assert result == [mock1, mock2]


class TestConverterWithReferencedConverter:
    """Tests for creating converters that reference other converters by ID."""

    @pytest.mark.asyncio
    async def test_create_converter_with_referenced_converter(self) -> None:
        """Test creating a converter that references another converter by ID."""
        service = ConverterService()

        mock_inner_class = MagicMock()
        mock_inner_instance = MagicMock()
        mock_inner_class.return_value = mock_inner_instance

        mock_outer_class = MagicMock()
        mock_outer_instance = MagicMock()
        mock_outer_class.return_value = mock_outer_instance

        def mock_get_class(converter_type: str) -> type:
            if converter_type == "OuterConverter":
                return mock_outer_class
            elif converter_type == "InnerConverter":
                return mock_inner_class
            raise ValueError(f"Unknown type: {converter_type}")

        with patch.object(service, "_get_converter_class", side_effect=mock_get_class):
            inner_result = await service.create_converter(CreateConverterRequest(type="InnerConverter", params={}))
            inner_id = inner_result.converter_id

            await service.create_converter(
                CreateConverterRequest(
                    type="OuterConverter",
                    params={"converter": {"converter_id": inner_id}},
                )
            )

            mock_outer_class.assert_called()
            call_kwargs = mock_outer_class.call_args[1]
            assert call_kwargs.get("converter") is mock_inner_instance

    @pytest.mark.asyncio
    async def test_create_converter_with_invalid_reference_raises(self) -> None:
        """Test that referencing a non-existent converter raises ValueError."""
        service = ConverterService()

        mock_class = MagicMock()
        with patch.object(service, "_get_converter_class", return_value=mock_class):
            with pytest.raises(ValueError, match="not found"):
                await service.create_converter(
                    CreateConverterRequest(
                        type="OuterConverter",
                        params={"converter": {"converter_id": "nonexistent"}},
                    )
                )


class TestConverterServiceSingleton:
    """Tests for get_converter_service singleton function."""

    def test_get_converter_service_returns_converter_service(self) -> None:
        """Test that get_converter_service returns a ConverterService instance."""
        import pyrit.backend.services.converter_service as module
        from pyrit.backend.services.converter_service import get_converter_service

        module._converter_service = None

        service = get_converter_service()
        assert isinstance(service, ConverterService)

    def test_get_converter_service_returns_same_instance(self) -> None:
        """Test that get_converter_service returns the same instance."""
        import pyrit.backend.services.converter_service as module
        from pyrit.backend.services.converter_service import get_converter_service

        module._converter_service = None

        service1 = get_converter_service()
        service2 = get_converter_service()
        assert service1 is service2
