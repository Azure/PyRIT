# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend target service.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from pyrit.backend.models.targets import CreateTargetRequest, TargetInstance
from pyrit.backend.services.target_service import TargetService


class TestTargetServiceInit:
    """Tests for TargetService initialization."""

    def test_init_creates_empty_instances_dict(self) -> None:
        """Test that service initializes with empty instances dictionary."""
        service = TargetService()
        assert service._instances == {}

    def test_init_creates_empty_target_objects_dict(self) -> None:
        """Test that service initializes with empty target objects dictionary."""
        service = TargetService()
        assert service._target_objects == {}


class TestListTargets:
    """Tests for TargetService.list_targets method."""

    @pytest.mark.asyncio
    async def test_list_targets_returns_empty_when_no_targets(self) -> None:
        """Test that list_targets returns empty list when no targets exist."""
        service = TargetService()

        result = await service.list_targets()

        assert result.items == []

    @pytest.mark.asyncio
    async def test_list_targets_returns_targets(self) -> None:
        """Test that list_targets returns existing targets."""
        service = TargetService()
        now = datetime.now(timezone.utc)

        service._instances["target-1"] = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            display_name="My Target",
            params={},
            created_at=now,
            source="user",
        )

        result = await service.list_targets()

        assert len(result.items) == 1
        assert result.items[0].target_id == "target-1"
        assert result.items[0].display_name == "My Target"

    @pytest.mark.asyncio
    async def test_list_targets_filters_by_source_user(self) -> None:
        """Test that list_targets filters by source='user'."""
        service = TargetService()
        now = datetime.now(timezone.utc)

        service._instances["target-1"] = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            params={},
            created_at=now,
            source="user",
        )
        service._instances["target-2"] = TargetInstance(
            target_id="target-2",
            type="TextTarget",
            params={},
            created_at=now,
            source="initializer",
        )

        result = await service.list_targets(source="user")

        assert len(result.items) == 1
        assert result.items[0].source == "user"

    @pytest.mark.asyncio
    async def test_list_targets_filters_by_source_initializer(self) -> None:
        """Test that list_targets filters by source='initializer'."""
        service = TargetService()
        now = datetime.now(timezone.utc)

        service._instances["target-1"] = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            params={},
            created_at=now,
            source="user",
        )
        service._instances["target-2"] = TargetInstance(
            target_id="target-2",
            type="TextTarget",
            params={},
            created_at=now,
            source="initializer",
        )

        result = await service.list_targets(source="initializer")

        assert len(result.items) == 1
        assert result.items[0].source == "initializer"


class TestGetTarget:
    """Tests for TargetService.get_target method."""

    @pytest.mark.asyncio
    async def test_get_target_returns_none_for_nonexistent(self) -> None:
        """Test that get_target returns None for non-existent target."""
        service = TargetService()

        result = await service.get_target("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_target_returns_target(self) -> None:
        """Test that get_target returns the target instance."""
        service = TargetService()
        now = datetime.now(timezone.utc)

        service._instances["target-1"] = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            display_name="Test Target",
            params={"key": "value"},
            created_at=now,
            source="user",
        )

        result = await service.get_target("target-1")

        assert result is not None
        assert result.target_id == "target-1"
        assert result.display_name == "Test Target"


class TestGetTargetObject:
    """Tests for TargetService.get_target_object method."""

    def test_get_target_object_returns_none_for_nonexistent(self) -> None:
        """Test that get_target_object returns None for non-existent target."""
        service = TargetService()

        result = service.get_target_object("nonexistent-id")

        assert result is None

    def test_get_target_object_returns_object(self) -> None:
        """Test that get_target_object returns the actual target object."""
        service = TargetService()
        mock_target = MagicMock()
        service._target_objects["target-1"] = mock_target

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

        # TextTarget should exist in pyrit.prompt_target
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

        # Use a target that doesn't require external dependencies
        request = CreateTargetRequest(
            type="TextTarget",
            display_name="My Text Target",
            params={},
        )

        result = await service.create_target(request)

        assert result.target_id is not None
        assert result.type == "TextTarget"
        assert result.display_name == "My Text Target"
        assert result.source == "user"

    @pytest.mark.asyncio
    async def test_create_target_stores_instance(self) -> None:
        """Test that create_target stores the instance."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            params={},
        )

        result = await service.create_target(request)

        assert result.target_id in service._instances
        assert result.target_id in service._target_objects

    @pytest.mark.asyncio
    async def test_create_target_filters_sensitive_params(self) -> None:
        """Test that create_target filters sensitive parameters."""
        service = TargetService()

        # Create a mock target class that has sensitive identifier fields
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
            # endpoint should remain
            assert result.params.get("endpoint") == "https://api.example.com"


class TestDeleteTarget:
    """Tests for TargetService.delete_target method."""

    @pytest.mark.asyncio
    async def test_delete_target_returns_false_for_nonexistent(self) -> None:
        """Test that delete_target returns False for non-existent target."""
        service = TargetService()

        result = await service.delete_target("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_target_deletes_target(self) -> None:
        """Test that delete_target removes the target."""
        service = TargetService()
        now = datetime.now(timezone.utc)

        service._instances["target-1"] = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            params={},
            created_at=now,
            source="user",
        )
        service._target_objects["target-1"] = MagicMock()

        result = await service.delete_target("target-1")

        assert result is True
        assert "target-1" not in service._instances
        assert "target-1" not in service._target_objects


class TestRegisterInitializerTarget:
    """Tests for TargetService.register_initializer_target method."""

    @pytest.mark.asyncio
    async def test_register_initializer_target_creates_instance(self) -> None:
        """Test that register_initializer_target creates an instance."""
        service = TargetService()
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"type": "MockTarget"}

        result = await service.register_initializer_target(
            target_type="MockTarget",
            target_obj=mock_target,
            display_name="Initializer Target",
        )

        assert result.target_id is not None
        assert result.type == "MockTarget"
        assert result.display_name == "Initializer Target"
        assert result.source == "initializer"

    @pytest.mark.asyncio
    async def test_register_initializer_target_stores_object(self) -> None:
        """Test that register_initializer_target stores the target object."""
        service = TargetService()
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {}

        result = await service.register_initializer_target(
            target_type="MockTarget",
            target_obj=mock_target,
        )

        assert service._target_objects[result.target_id] is mock_target


class TestTargetServiceSingleton:
    """Tests for get_target_service singleton function."""

    def test_get_target_service_returns_target_service(self) -> None:
        """Test that get_target_service returns a TargetService instance."""
        # Reset singleton for clean test
        import pyrit.backend.services.target_service as module
        from pyrit.backend.services.target_service import get_target_service

        module._target_service = None

        service = get_target_service()
        assert isinstance(service, TargetService)

    def test_get_target_service_returns_same_instance(self) -> None:
        """Test that get_target_service returns the same instance."""
        # Reset singleton for clean test
        import pyrit.backend.services.target_service as module
        from pyrit.backend.services.target_service import get_target_service

        module._target_service = None

        service1 = get_target_service()
        service2 = get_target_service()
        assert service1 is service2
