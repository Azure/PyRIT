# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend target service.
"""

from unittest.mock import MagicMock, patch

import pytest

from pyrit.backend.models.targets import CreateTargetRequest
from pyrit.backend.services.target_service import TargetService
from pyrit.registry.instance_registries import TargetRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the TargetRegistry singleton before each test."""
    TargetRegistry.reset_instance()
    yield
    TargetRegistry.reset_instance()


class TestListTargets:
    """Tests for TargetService.list_targets method."""

    @pytest.mark.asyncio
    async def test_list_targets_returns_empty_when_no_targets(self) -> None:
        """Test that list_targets returns empty list when no targets exist."""
        service = TargetService()

        result = await service.list_targets()

        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_targets_returns_targets_from_registry(self) -> None:
        """Test that list_targets returns targets from registry."""
        service = TargetService()

        # Register a mock target
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"__type__": "MockTarget", "endpoint": "http://test"}
        service._registry.register_instance(mock_target, name="target-1")

        result = await service.list_targets()

        assert len(result.items) == 1
        assert result.items[0].target_id == "target-1"
        assert result.items[0].type == "MockTarget"


class TestGetTarget:
    """Tests for TargetService.get_target method."""

    @pytest.mark.asyncio
    async def test_get_target_returns_none_for_nonexistent(self) -> None:
        """Test that get_target returns None for non-existent target."""
        service = TargetService()

        result = await service.get_target("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_target_returns_target_from_registry(self) -> None:
        """Test that get_target returns target built from registry object."""
        service = TargetService()

        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"__type__": "MockTarget"}
        service._registry.register_instance(mock_target, name="target-1")

        result = await service.get_target("target-1")

        assert result is not None
        assert result.target_id == "target-1"
        assert result.type == "MockTarget"


class TestGetTargetObject:
    """Tests for TargetService.get_target_object method."""

    def test_get_target_object_returns_none_for_nonexistent(self) -> None:
        """Test that get_target_object returns None for non-existent target."""
        service = TargetService()

        result = service.get_target_object("nonexistent-id")

        assert result is None

    def test_get_target_object_returns_object_from_registry(self) -> None:
        """Test that get_target_object returns the actual target object."""
        service = TargetService()
        mock_target = MagicMock()
        service._registry.register_instance(mock_target, name="target-1")

        result = service.get_target_object("target-1")

        assert result is mock_target


class TestGetTargetClass:
    """Tests for TargetService._get_target_class method."""

    def test_get_target_class_raises_for_invalid_type(self) -> None:
        """Test that _get_target_class raises ValueError for invalid type."""
        service = TargetService()

        with pytest.raises(ValueError, match="not found"):
            service._get_target_class("NonExistentTarget")

    def test_get_target_class_finds_text_target(self) -> None:
        """Test that _get_target_class finds TextTarget."""
        service = TargetService()

        result = service._get_target_class("TextTarget")

        assert result is not None
        assert "TextTarget" in result.__name__


class TestCreateTarget:
    """Tests for TargetService.create_target method."""

    @pytest.mark.asyncio
    async def test_create_target_raises_for_invalid_type(self) -> None:
        """Test that create_target raises for invalid target type."""
        service = TargetService()

        request = CreateTargetRequest(
            type="NonExistentTarget",
            params={},
        )

        with pytest.raises(ValueError, match="not found"):
            await service.create_target(request)

    @pytest.mark.asyncio
    async def test_create_target_success(self) -> None:
        """Test successful target creation."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            display_name="My Text Target",
            params={},
        )

        result = await service.create_target(request)

        assert result.target_id is not None
        assert result.type == "TextTarget"
        assert result.display_name == "My Text Target"

    @pytest.mark.asyncio
    async def test_create_target_registers_in_registry(self) -> None:
        """Test that create_target registers object in registry."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            params={},
        )

        result = await service.create_target(request)

        # Object should be retrievable from registry
        target_obj = service.get_target_object(result.target_id)
        assert target_obj is not None

    @pytest.mark.asyncio
    async def test_create_target_filters_sensitive_params(self) -> None:
        """Test that create_target filters sensitive parameters."""
        service = TargetService()

        mock_target_class = MagicMock()
        mock_target_instance = MagicMock()
        mock_target_instance.get_identifier.return_value = {
            "type": "MockTarget",
            "api_key": "secret-key",
            "endpoint": "https://api.example.com",
        }
        mock_target_class.return_value = mock_target_instance

        with patch.object(service, "_get_target_class", return_value=mock_target_class):
            request = CreateTargetRequest(
                type="MockTarget",
                params={},
            )

            result = await service.create_target(request)

            # api_key should be filtered out
            assert "api_key" not in result.params
            assert result.params.get("endpoint") == "https://api.example.com"


class TestTargetServiceSingleton:
    """Tests for get_target_service singleton function."""

    def test_get_target_service_returns_target_service(self) -> None:
        """Test that get_target_service returns a TargetService instance."""
        import pyrit.backend.services.target_service as module
        from pyrit.backend.services.target_service import get_target_service

        module._target_service = None

        service = get_target_service()
        assert isinstance(service, TargetService)

    def test_get_target_service_returns_same_instance(self) -> None:
        """Test that get_target_service returns the same instance."""
        import pyrit.backend.services.target_service as module
        from pyrit.backend.services.target_service import get_target_service

        module._target_service = None

        service1 = get_target_service()
        service2 = get_target_service()
        assert service1 is service2
