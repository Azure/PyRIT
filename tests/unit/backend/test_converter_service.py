# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend converter service.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterPreviewRequest,
    CreateConverterRequest,
    InlineConverterConfig,
)
from pyrit.backend.services.converter_service import ConverterService
from pyrit.models import PromptDataType


class TestConverterServiceInit:
    """Tests for ConverterService initialization."""

    def test_init_creates_empty_instances_dict(self) -> None:
        """Test that service initializes with empty instances dictionary."""
        service = ConverterService()
        assert service._instances == {}

    def test_init_creates_empty_converter_objects_dict(self) -> None:
        """Test that service initializes with empty converter objects dictionary."""
        service = ConverterService()
        assert service._converter_objects == {}


class TestListConverters:
    """Tests for ConverterService.list_converters method."""

    @pytest.mark.asyncio
    async def test_list_converters_returns_empty_when_no_converters(self) -> None:
        """Test that list_converters returns empty list when no converters exist."""
        service = ConverterService()

        result = await service.list_converters()

        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_converters_returns_converters(self) -> None:
        """Test that list_converters returns existing converters."""
        service = ConverterService()
        now = datetime.now(timezone.utc)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="Base64Converter",
            display_name="My Converter",
            params={},
            created_at=now,
            source="user",
        )

        result = await service.list_converters()

        assert len(result.items) == 1
        assert result.items[0].converter_id == "conv-1"
        assert result.items[0].display_name == "My Converter"

    @pytest.mark.asyncio
    async def test_list_converters_filters_by_source_user(self) -> None:
        """Test that list_converters filters by source='user'."""
        service = ConverterService()
        now = datetime.now(timezone.utc)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="Base64Converter",
            params={},
            created_at=now,
            source="user",
        )
        service._instances["conv-2"] = ConverterInstance(
            converter_id="conv-2",
            type="Base64Converter",
            params={},
            created_at=now,
            source="initializer",
        )

        result = await service.list_converters(source="user")

        assert len(result.items) == 1
        assert result.items[0].source == "user"

    @pytest.mark.asyncio
    async def test_list_converters_filters_by_source_initializer(self) -> None:
        """Test that list_converters filters by source='initializer'."""
        service = ConverterService()
        now = datetime.now(timezone.utc)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="Base64Converter",
            params={},
            created_at=now,
            source="user",
        )
        service._instances["conv-2"] = ConverterInstance(
            converter_id="conv-2",
            type="Base64Converter",
            params={},
            created_at=now,
            source="initializer",
        )

        result = await service.list_converters(source="initializer")

        assert len(result.items) == 1
        assert result.items[0].source == "initializer"


class TestGetConverter:
    """Tests for ConverterService.get_converter method."""

    @pytest.mark.asyncio
    async def test_get_converter_returns_none_for_nonexistent(self) -> None:
        """Test that get_converter returns None for non-existent converter."""
        service = ConverterService()

        result = await service.get_converter("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_converter_returns_converter(self) -> None:
        """Test that get_converter returns the converter instance."""
        service = ConverterService()
        now = datetime.now(timezone.utc)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="Base64Converter",
            display_name="Test Converter",
            params={"key": "value"},
            created_at=now,
            source="user",
        )

        result = await service.get_converter("conv-1")

        assert result is not None
        assert result.converter_id == "conv-1"
        assert result.display_name == "Test Converter"


class TestGetConverterObject:
    """Tests for ConverterService.get_converter_object method."""

    def test_get_converter_object_returns_none_for_nonexistent(self) -> None:
        """Test that get_converter_object returns None for non-existent converter."""
        service = ConverterService()

        result = service.get_converter_object("nonexistent-id")

        assert result is None

    def test_get_converter_object_returns_object(self) -> None:
        """Test that get_converter_object returns the actual converter object."""
        service = ConverterService()
        mock_converter = MagicMock()
        service._converter_objects["conv-1"] = mock_converter

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

        # base64 should resolve to Base64Converter
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
        assert result.source == "user"

    @pytest.mark.asyncio
    async def test_create_converter_stores_instance(self) -> None:
        """Test that create_converter stores the instance."""
        service = ConverterService()

        request = CreateConverterRequest(
            type="Base64Converter",
            params={},
        )

        result = await service.create_converter(request)

        assert result.converter_id in service._instances
        assert result.converter_id in service._converter_objects


class TestDeleteConverter:
    """Tests for ConverterService.delete_converter method."""

    @pytest.mark.asyncio
    async def test_delete_converter_returns_false_for_nonexistent(self) -> None:
        """Test that delete_converter returns False for non-existent converter."""
        service = ConverterService()

        result = await service.delete_converter("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_converter_deletes_converter(self) -> None:
        """Test that delete_converter removes the converter."""
        service = ConverterService()
        now = datetime.now(timezone.utc)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="Base64Converter",
            params={},
            created_at=now,
            source="user",
        )
        service._converter_objects["conv-1"] = MagicMock()

        result = await service.delete_converter("conv-1")

        assert result is True
        assert "conv-1" not in service._instances
        assert "conv-1" not in service._converter_objects


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
        now = datetime.now(timezone.utc)

        # Create a mock converter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.output_text = "encoded_value"
        mock_result.output_type = "text"
        mock_converter.convert_async = AsyncMock(return_value=mock_result)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="MockConverter",
            params={},
            created_at=now,
            source="user",
        )
        service._converter_objects["conv-1"] = mock_converter

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
    async def test_preview_conversion_with_inline_converters(self) -> None:
        """Test preview with inline converter configs."""
        service = ConverterService()

        request = ConverterPreviewRequest(
            original_value="test",
            original_value_data_type="text",
            converters=[
                InlineConverterConfig(type="Base64Converter", params={}),
            ],
        )

        result = await service.preview_conversion(request)

        assert result.original_value == "test"
        assert result.converted_value is not None
        assert len(result.steps) == 1
        # Base64 of "test" should be different from "test"
        assert result.converted_value != "test"

    @pytest.mark.asyncio
    async def test_preview_conversion_chains_multiple_converters(self) -> None:
        """Test that preview chains multiple converters."""
        service = ConverterService()
        now = datetime.now(timezone.utc)

        # Create two mock converters
        mock_converter1 = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.output_text = "step1_output"
        mock_result1.output_type = "text"
        mock_converter1.convert_async = AsyncMock(return_value=mock_result1)

        mock_converter2 = MagicMock()
        mock_result2 = MagicMock()
        mock_result2.output_text = "step2_output"
        mock_result2.output_type = "text"
        mock_converter2.convert_async = AsyncMock(return_value=mock_result2)

        service._instances["conv-1"] = ConverterInstance(
            converter_id="conv-1",
            type="MockConverter1",
            params={},
            created_at=now,
            source="user",
        )
        service._converter_objects["conv-1"] = mock_converter1

        service._instances["conv-2"] = ConverterInstance(
            converter_id="conv-2",
            type="MockConverter2",
            params={},
            created_at=now,
            source="user",
        )
        service._converter_objects["conv-2"] = mock_converter2

        request = ConverterPreviewRequest(
            original_value="input",
            original_value_data_type="text",
            converter_ids=["conv-1", "conv-2"],
        )

        result = await service.preview_conversion(request)

        assert result.converted_value == "step2_output"
        assert len(result.steps) == 2
        # Second converter should receive output from first
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
        service._converter_objects["conv-1"] = mock1
        service._converter_objects["conv-2"] = mock2

        result = service.get_converter_objects_for_ids(["conv-1", "conv-2"])

        assert result == [mock1, mock2]


class TestInstantiateInlineConverters:
    """Tests for ConverterService.instantiate_inline_converters method."""

    def test_instantiate_inline_converters_creates_objects(self) -> None:
        """Test that inline converters are instantiated."""
        service = ConverterService()

        configs = [
            InlineConverterConfig(type="Base64Converter", params={}),
        ]

        result = service.instantiate_inline_converters(configs)

        assert len(result) == 1
        # Verify it's a real converter object
        assert hasattr(result[0], "convert_async")

    def test_instantiate_inline_converters_raises_for_invalid_type(self) -> None:
        """Test that invalid type raises ValueError."""
        service = ConverterService()

        configs = [
            InlineConverterConfig(type="NonExistentConverter", params={}),
        ]

        with pytest.raises(ValueError, match="not found"):
            service.instantiate_inline_converters(configs)


class TestNestedConverterCreation:
    """Tests for nested converter creation."""

    @pytest.mark.asyncio
    async def test_create_converter_with_nested_converter(self) -> None:
        """Test creating a converter with a nested converter config."""
        service = ConverterService()

        # Mock the parent converter class that accepts a 'converter' param
        mock_parent_class = MagicMock()
        mock_parent_instance = MagicMock()
        mock_parent_class.return_value = mock_parent_instance

        mock_child_class = MagicMock()
        mock_child_instance = MagicMock()
        mock_child_class.return_value = mock_child_instance

        def mock_get_class(converter_type: str) -> type:
            if converter_type == "ParentConverter":
                return mock_parent_class
            elif converter_type == "ChildConverter":
                return mock_child_class
            raise ValueError(f"Unknown type: {converter_type}")

        with patch.object(service, "_get_converter_class", side_effect=mock_get_class):
            request = CreateConverterRequest(
                type="ParentConverter",
                params={
                    "converter": {
                        "type": "ChildConverter",
                        "params": {},
                    },
                },
            )

            result = await service.create_converter(request)

            # Parent should be created with child converter object
            mock_parent_class.assert_called()
            # The call should have received the child instance, not the dict
            call_kwargs = mock_parent_class.call_args[1]
            assert call_kwargs.get("converter") is mock_child_instance


class TestConverterServiceSingleton:
    """Tests for get_converter_service singleton function."""

    def test_get_converter_service_returns_converter_service(self) -> None:
        """Test that get_converter_service returns a ConverterService instance."""
        from pyrit.backend.services.converter_service import get_converter_service

        # Reset singleton for clean test
        import pyrit.backend.services.converter_service as module
        module._converter_service = None

        service = get_converter_service()
        assert isinstance(service, ConverterService)

    def test_get_converter_service_returns_same_instance(self) -> None:
        """Test that get_converter_service returns the same instance."""
        from pyrit.backend.services.converter_service import get_converter_service

        # Reset singleton for clean test
        import pyrit.backend.services.converter_service as module
        module._converter_service = None

        service1 = get_converter_service()
        service2 = get_converter_service()
        assert service1 is service2
