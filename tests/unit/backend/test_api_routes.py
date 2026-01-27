# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend API routes.
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pyrit.backend.main import app
from pyrit.backend.models.attacks import (
    AttackDetail,
    AttackListResponse,
    AttackSummary,
    CreateAttackResponse,
    Message,
    MessagePiece,
    SendMessageResponse,
)
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterInstanceListResponse,
    ConverterPreviewResponse,
    CreateConverterResponse,
    PreviewStep,
)
from pyrit.backend.models.targets import (
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ============================================================================
# Attack Routes Tests
# ============================================================================


class TestAttackRoutes:
    """Tests for attack API routes."""

    def test_list_attacks_returns_empty_list(self, client: TestClient) -> None:
        """Test that list attacks returns empty list initially."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_attacks = AsyncMock(
                return_value=AttackListResponse(
                    items=[],
                    pagination=PaginationInfo(limit=20, has_more=False),
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/attacks")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["items"] == []

    def test_list_attacks_with_filters(self, client: TestClient) -> None:
        """Test that list attacks accepts filter parameters."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_attacks = AsyncMock(
                return_value=AttackListResponse(
                    items=[],
                    pagination=PaginationInfo(limit=10, has_more=False),
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get(
                "/api/attacks",
                params={"target_id": "t1", "outcome": "success", "limit": 10},
            )

            assert response.status_code == status.HTTP_200_OK
            mock_service.list_attacks.assert_called_once_with(
                target_id="t1",
                outcome="success",
                limit=10,
                cursor=None,
            )

    def test_create_attack_success(self, client: TestClient) -> None:
        """Test successful attack creation."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_attack = AsyncMock(
                return_value=CreateAttackResponse(
                    attack_id="attack-1",
                    name="Test Attack",
                    target_id="target-1",
                    target_type="TextTarget",
                    outcome=None,
                    prepended_conversation=[],
                    created_at=now,
                    updated_at=now,
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks",
                json={"target_id": "target-1", "name": "Test Attack"},
            )

            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["attack_id"] == "attack-1"

    def test_create_attack_target_not_found(self, client: TestClient) -> None:
        """Test attack creation with non-existent target."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_attack = AsyncMock(
                side_effect=ValueError("Target not found")
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks",
                json={"target_id": "nonexistent"},
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_attack_success(self, client: TestClient) -> None:
        """Test getting an attack by ID."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_attack = AsyncMock(
                return_value=AttackDetail(
                    attack_id="attack-1",
                    name="Test",
                    target_id="target-1",
                    target_type="TextTarget",
                    outcome=None,
                    prepended_conversation=[],
                    messages=[],
                    converter_ids=[],
                    created_at=now,
                    updated_at=now,
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/attacks/attack-1")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["attack_id"] == "attack-1"

    def test_get_attack_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent attack."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_attack = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/api/attacks/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_attack_success(self, client: TestClient) -> None:
        """Test updating an attack's outcome."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.update_attack = AsyncMock(
                return_value=AttackDetail(
                    attack_id="attack-1",
                    target_id="target-1",
                    target_type="TextTarget",
                    outcome="success",
                    prepended_conversation=[],
                    messages=[],
                    converter_ids=[],
                    created_at=now,
                    updated_at=now,
                )
            )
            mock_get_service.return_value = mock_service

            response = client.patch(
                "/api/attacks/attack-1",
                json={"outcome": "success"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["outcome"] == "success"

    def test_delete_attack_success(self, client: TestClient) -> None:
        """Test deleting an attack."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_attack = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            response = client.delete("/api/attacks/attack-1")

            assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_attack_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent attack."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_attack = AsyncMock(return_value=False)
            mock_get_service.return_value = mock_service

            response = client.delete("/api/attacks/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_send_message_success(self, client: TestClient) -> None:
        """Test sending a message in an attack."""
        now = datetime.now(timezone.utc)

        user_msg = Message(
            message_id="msg-1",
            turn_number=1,
            role="user",
            pieces=[
                MessagePiece(
                    piece_id="piece-1",
                    data_type="text",
                    converted_value="Hello",
                    scores=[],
                )
            ],
            created_at=now,
        )
        assistant_msg = Message(
            message_id="msg-2",
            turn_number=2,
            role="assistant",
            pieces=[
                MessagePiece(
                    piece_id="piece-2",
                    data_type="text",
                    converted_value="Hi there!",
                    scores=[],
                )
            ],
            created_at=now,
        )
        summary = AttackSummary(
            attack_id="attack-1",
            target_id="target-1",
            target_type="TextTarget",
            message_count=2,
            created_at=now,
            updated_at=now,
        )

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.send_message = AsyncMock(
                return_value=SendMessageResponse(
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                    attack_summary=summary,
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks/attack-1/messages",
                json={"pieces": [{"content": "Hello"}]},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["user_message"]["pieces"][0]["converted_value"] == "Hello"


# ============================================================================
# Target Routes Tests
# ============================================================================


class TestTargetRoutes:
    """Tests for target API routes."""

    def test_list_targets_returns_empty_list(self, client: TestClient) -> None:
        """Test that list targets returns empty list initially."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_targets = AsyncMock(
                return_value=TargetListResponse(items=[])
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/targets")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["items"] == []

    def test_list_targets_with_source_filter(self, client: TestClient) -> None:
        """Test that list targets accepts source filter."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_targets = AsyncMock(
                return_value=TargetListResponse(items=[])
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/targets", params={"source": "user"})

            assert response.status_code == status.HTTP_200_OK
            mock_service.list_targets.assert_called_once_with(source="user")

    def test_create_target_success(self, client: TestClient) -> None:
        """Test successful target creation."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_target = AsyncMock(
                return_value=CreateTargetResponse(
                    target_id="target-1",
                    type="TextTarget",
                    display_name="My Target",
                    params={},
                    created_at=now,
                    source="user",
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/targets",
                json={"type": "TextTarget", "display_name": "My Target", "params": {}},
            )

            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["target_id"] == "target-1"

    def test_create_target_invalid_type(self, client: TestClient) -> None:
        """Test target creation with invalid type."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_target = AsyncMock(
                side_effect=ValueError("Target type not found")
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/targets",
                json={"type": "InvalidTarget", "params": {}},
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_get_target_success(self, client: TestClient) -> None:
        """Test getting a target by ID."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_target = AsyncMock(
                return_value=TargetInstance(
                    target_id="target-1",
                    type="TextTarget",
                    params={},
                    created_at=now,
                    source="user",
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/targets/target-1")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["target_id"] == "target-1"

    def test_get_target_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent target."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_target = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/api/targets/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_target_success(self, client: TestClient) -> None:
        """Test deleting a target."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_target = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            response = client.delete("/api/targets/target-1")

            assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_target_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent target."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_target = AsyncMock(return_value=False)
            mock_get_service.return_value = mock_service

            response = client.delete("/api/targets/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Converter Routes Tests
# ============================================================================


class TestConverterRoutes:
    """Tests for converter API routes."""

    def test_list_converter_types(self, client: TestClient) -> None:
        """Test listing converter types from registry."""
        with patch("pyrit.backend.routes.converters.get_registry_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_converters.return_value = []
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters/types")

            assert response.status_code == status.HTTP_200_OK

    def test_list_converter_instances(self, client: TestClient) -> None:
        """Test listing converter instances."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_converters = AsyncMock(
                return_value=ConverterInstanceListResponse(items=[])
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters/instances")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["items"] == []

    def test_create_converter_instance_success(self, client: TestClient) -> None:
        """Test successful converter instance creation."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_converter = AsyncMock(
                return_value=CreateConverterResponse(
                    converter_id="conv-1",
                    type="Base64Converter",
                    display_name="My Base64",
                    params={},
                    created_at=now,
                    source="user",
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters/instances",
                json={"type": "Base64Converter", "display_name": "My Base64", "params": {}},
            )

            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["converter_id"] == "conv-1"

    def test_create_converter_instance_invalid_type(self, client: TestClient) -> None:
        """Test converter creation with invalid type."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_converter = AsyncMock(
                side_effect=ValueError("Converter type not found")
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters/instances",
                json={"type": "InvalidConverter", "params": {}},
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_get_converter_instance_success(self, client: TestClient) -> None:
        """Test getting a converter instance by ID."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_converter = AsyncMock(
                return_value=ConverterInstance(
                    converter_id="conv-1",
                    type="Base64Converter",
                    params={},
                    created_at=now,
                    source="user",
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters/instances/conv-1")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["converter_id"] == "conv-1"

    def test_get_converter_instance_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent converter instance."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_converter = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters/instances/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_converter_instance_success(self, client: TestClient) -> None:
        """Test deleting a converter instance."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_converter = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            response = client.delete("/api/converters/instances/conv-1")

            assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_converter_instance_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent converter instance."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.delete_converter = AsyncMock(return_value=False)
            mock_get_service.return_value = mock_service

            response = client.delete("/api/converters/instances/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_preview_conversion_success(self, client: TestClient) -> None:
        """Test previewing a conversion."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.preview_conversion = AsyncMock(
                return_value=ConverterPreviewResponse(
                    original_value="test",
                    original_value_data_type="text",
                    converted_value="dGVzdA==",
                    converted_value_data_type="text",
                    steps=[
                        PreviewStep(
                            converter_id=None,
                            converter_type="Base64Converter",
                            input_value="test",
                            input_data_type="text",
                            output_value="dGVzdA==",
                            output_data_type="text",
                        )
                    ],
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters/preview",
                json={
                    "original_value": "test",
                    "original_value_data_type": "text",
                    "converters": [{"type": "Base64Converter", "params": {}}],
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["converted_value"] == "dGVzdA=="
            assert len(data["steps"]) == 1
