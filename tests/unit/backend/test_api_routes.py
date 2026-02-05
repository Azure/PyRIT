# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend API routes.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pyrit.backend.main import app
from pyrit.backend.models.attacks import (
    AddMessageResponse,
    AttackListResponse,
    AttackMessagesResponse,
    AttackSummary,
    CreateAttackResponse,
    Message,
    MessagePiece,
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
                    pagination=PaginationInfo(limit=20, has_more=False, next_cursor=None, prev_cursor=None),
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
                    pagination=PaginationInfo(limit=10, has_more=False, next_cursor=None, prev_cursor=None),
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
                name=None,
                labels=None,
                min_turns=None,
                max_turns=None,
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
                    created_at=now,
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
            mock_service.create_attack = AsyncMock(side_effect=ValueError("Target not found"))
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
                return_value=AttackSummary(
                    attack_id="attack-1",
                    name="Test",
                    target_id="target-1",
                    target_type="TextTarget",
                    outcome=None,
                    last_message_preview=None,
                    message_count=0,
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
                return_value=AttackSummary(
                    attack_id="attack-1",
                    name=None,
                    target_id="target-1",
                    target_type="TextTarget",
                    outcome="success",
                    last_message_preview=None,
                    message_count=0,
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

    def test_add_message_success(self, client: TestClient) -> None:
        """Test adding a message to an attack."""
        now = datetime.now(timezone.utc)

        attack_summary = AttackSummary(
            attack_id="attack-1",
            name=None,
            target_id="target-1",
            target_type="TextTarget",
            outcome=None,
            last_message_preview=None,
            message_count=2,
            created_at=now,
            updated_at=now,
        )

        attack_messages = AttackMessagesResponse(
            attack_id="attack-1",
            messages=[
                Message(
                    message_id="msg-1",
                    turn_number=1,
                    role="user",
                    pieces=[
                        MessagePiece(
                            piece_id="piece-1",
                            converted_value="Hello",
                        )
                    ],
                    created_at=now,
                ),
                Message(
                    message_id="msg-2",
                    turn_number=2,
                    role="assistant",
                    pieces=[
                        MessagePiece(
                            piece_id="piece-2",
                            converted_value="Hi there!",
                        )
                    ],
                    created_at=now,
                ),
            ],
        )

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.add_message = AsyncMock(
                return_value=AddMessageResponse(
                    attack=attack_summary,
                    messages=attack_messages,
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks/attack-1/messages",
                json={"pieces": [{"original_value": "Hello"}]},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["messages"]["messages"]) == 2

    def test_update_attack_not_found(self, client: TestClient) -> None:
        """Test updating a non-existent attack returns 404."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.update_attack = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.patch(
                "/api/attacks/nonexistent",
                json={"outcome": "success"},
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_add_message_attack_not_found(self, client: TestClient) -> None:
        """Test adding message to non-existent attack returns 404."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.add_message = AsyncMock(side_effect=ValueError("Attack 'nonexistent' not found"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks/nonexistent/messages",
                json={"pieces": [{"original_value": "Hello"}]},
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_add_message_target_not_found(self, client: TestClient) -> None:
        """Test adding message when target object not found returns 404."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.add_message = AsyncMock(side_effect=ValueError("Target object for 'target-1' not found"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks/attack-1/messages",
                json={"pieces": [{"original_value": "Hello"}]},
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_add_message_bad_request(self, client: TestClient) -> None:
        """Test adding message with invalid request returns 400."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.add_message = AsyncMock(side_effect=ValueError("Invalid message format"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks/attack-1/messages",
                json={"pieces": [{"original_value": "Hello"}]},
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_add_message_internal_error(self, client: TestClient) -> None:
        """Test adding message when internal error occurs returns 500."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.add_message = AsyncMock(side_effect=RuntimeError("Unexpected internal error"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/attacks/attack-1/messages",
                json={"pieces": [{"original_value": "Hello"}]},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_get_attack_messages_success(self, client: TestClient) -> None:
        """Test getting attack messages."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_attack_messages = AsyncMock(
                return_value=AttackMessagesResponse(
                    attack_id="attack-1",
                    messages=[
                        Message(
                            message_id="msg-1",
                            turn_number=1,
                            role="user",
                            pieces=[MessagePiece(piece_id="p1", converted_value="Hello")],
                            created_at=now,
                        )
                    ],
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/attacks/attack-1/messages")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["attack_id"] == "attack-1"
            assert len(data["messages"]) == 1

    def test_get_attack_messages_not_found(self, client: TestClient) -> None:
        """Test getting messages for non-existent attack returns 404."""
        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_attack_messages = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/api/attacks/nonexistent/messages")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_attacks_with_labels(self, client: TestClient) -> None:
        """Test listing attacks with label filters."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.attacks.get_attack_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_attacks = AsyncMock(
                return_value=AttackListResponse(
                    items=[
                        AttackSummary(
                            attack_id="attack-1",
                            name=None,
                            target_id="target-1",
                            target_type="TextTarget",
                            outcome=None,
                            last_message_preview=None,
                            message_count=0,
                            labels={"env": "prod"},
                            created_at=now,
                            updated_at=now,
                        )
                    ],
                    pagination=PaginationInfo(limit=20, has_more=False, next_cursor=None, prev_cursor=None),
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/attacks?label=env:prod&label=team:red")

            assert response.status_code == status.HTTP_200_OK
            # Verify labels were parsed and passed to service
            mock_service.list_attacks.assert_called_once()
            call_kwargs = mock_service.list_attacks.call_args[1]
            assert call_kwargs["labels"] == {"env": "prod", "team": "red"}


# ============================================================================
# Target Routes Tests
# ============================================================================


class TestTargetRoutes:
    """Tests for target API routes."""

    def test_list_targets_returns_empty_list(self, client: TestClient) -> None:
        """Test that list targets returns empty list initially."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_targets = AsyncMock(return_value=TargetListResponse(items=[]))
            mock_get_service.return_value = mock_service

            response = client.get("/api/targets")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["items"] == []

    def test_list_targets_with_source_filter(self, client: TestClient) -> None:
        """Test that list targets accepts source filter."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_targets = AsyncMock(return_value=TargetListResponse(items=[]))
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
            mock_service.create_target = AsyncMock(side_effect=ValueError("Target type not found"))
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
                    display_name=None,
                    params={},
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

    def test_create_target_internal_error(self, client: TestClient) -> None:
        """Test target creation with internal error returns 500."""
        with patch("pyrit.backend.routes.targets.get_target_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_target = AsyncMock(side_effect=RuntimeError("Unexpected internal error"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/targets",
                json={"type": "TextTarget", "params": {}},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ============================================================================
# Converter Routes Tests
# ============================================================================


class TestConverterRoutes:
    """Tests for converter API routes."""

    def test_list_converters(self, client: TestClient) -> None:
        """Test listing converter instances."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_converters = AsyncMock(return_value=ConverterInstanceListResponse(items=[]))
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["items"] == []

    def test_create_converter_success(self, client: TestClient) -> None:
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
                )
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters",
                json={"type": "Base64Converter", "display_name": "My Base64", "params": {}},
            )

            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["converter_id"] == "conv-1"

    def test_create_converter_invalid_type(self, client: TestClient) -> None:
        """Test converter creation with invalid type."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_converter = AsyncMock(side_effect=ValueError("Converter type not found"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters",
                json={"type": "InvalidConverter", "params": {}},
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_get_converter_success(self, client: TestClient) -> None:
        """Test getting a converter instance by ID."""
        now = datetime.now(timezone.utc)

        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_converter = AsyncMock(
                return_value=ConverterInstance(
                    converter_id="conv-1",
                    type="Base64Converter",
                    display_name=None,
                    params={},
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters/conv-1")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["converter_id"] == "conv-1"

    def test_get_converter_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent converter instance."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_converter = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/api/converters/nonexistent")

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
                            converter_id="conv-1",
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
                    "converter_ids": ["conv-1"],
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["converted_value"] == "dGVzdA=="
            assert len(data["steps"]) == 1

    def test_create_converter_internal_error(self, client: TestClient) -> None:
        """Test converter creation with internal error returns 500."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.create_converter = AsyncMock(side_effect=RuntimeError("Unexpected internal error"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters",
                json={"type": "Base64Converter", "params": {}},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_preview_conversion_bad_request(self, client: TestClient) -> None:
        """Test preview conversion with invalid converter ID returns 400."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.preview_conversion = AsyncMock(
                side_effect=ValueError("Converter instance 'nonexistent' not found")
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters/preview",
                json={
                    "original_value": "test",
                    "original_value_data_type": "text",
                    "converter_ids": ["nonexistent"],
                },
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_preview_conversion_internal_error(self, client: TestClient) -> None:
        """Test preview conversion with internal error returns 500."""
        with patch("pyrit.backend.routes.converters.get_converter_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.preview_conversion = AsyncMock(side_effect=RuntimeError("Converter execution failed"))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/converters/preview",
                json={
                    "original_value": "test",
                    "original_value_data_type": "text",
                    "converter_ids": ["conv-1"],
                },
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ============================================================================
# Version Routes Tests
# ============================================================================


class TestVersionRoutes:
    """Tests for version API routes."""

    def test_get_version(self, client: TestClient) -> None:
        """Test getting version information."""
        response = client.get("/api/version")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "version" in data
        assert "display" in data

    def test_get_version_with_build_info(self, client: TestClient) -> None:
        """Test getting version with build info from Docker."""
        # Create a temp file to simulate Docker build info
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "source": "git",
                    "commit": "abc123",
                    "modified": False,
                    "display": "1.0.0-test",
                },
                f,
            )
            temp_path = f.name

        try:
            with patch("pyrit.backend.routes.version.Path") as mock_path_class:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                mock_path_class.return_value = mock_path_instance

                # Mock open to return our temp file content
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                        {
                            "source": "git",
                            "commit": "abc123",
                            "modified": False,
                            "display": "1.0.0-test",
                        }
                    )

                    response = client.get("/api/version")

            assert response.status_code == status.HTTP_200_OK
        finally:
            os.unlink(temp_path)


# ============================================================================
# Health Routes Tests
# ============================================================================


class TestHealthRoutes:
    """Tests for health check API routes."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint returns ok."""
        response = client.get("/api/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"


# ============================================================================
# Labels Routes Tests
# ============================================================================


class TestLabelsRoutes:
    """Tests for labels API routes."""

    def test_get_labels_for_attacks(self, client: TestClient) -> None:
        """Test getting labels from attack results."""
        mock_attack_result = MagicMock()
        mock_attack_result.metadata = {"env": "prod", "team": "red", "created_at": "2024-01-01"}

        with patch("pyrit.backend.routes.labels.CentralMemory") as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory.get_attack_results.return_value = [mock_attack_result]
            mock_memory_class.get_memory_instance.return_value = mock_memory

            response = client.get("/api/labels?source=attacks")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["source"] == "attacks"
            # env and team should be included, created_at should be excluded
            assert "env" in data["labels"]
            assert "team" in data["labels"]
            assert "created_at" not in data["labels"]

    def test_get_labels_empty(self, client: TestClient) -> None:
        """Test getting labels when no attack results exist."""
        with patch("pyrit.backend.routes.labels.CentralMemory") as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory.get_attack_results.return_value = []
            mock_memory_class.get_memory_instance.return_value = mock_memory

            response = client.get("/api/labels?source=attacks")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["source"] == "attacks"
            assert data["labels"] == {}

    def test_get_labels_multiple_values(self, client: TestClient) -> None:
        """Test getting labels with multiple values per key."""
        mock_ar1 = MagicMock()
        mock_ar1.metadata = {"env": "prod"}
        mock_ar2 = MagicMock()
        mock_ar2.metadata = {"env": "staging"}
        mock_ar3 = MagicMock()
        mock_ar3.metadata = {"env": "prod", "team": "blue"}

        with patch("pyrit.backend.routes.labels.CentralMemory") as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory.get_attack_results.return_value = [mock_ar1, mock_ar2, mock_ar3]
            mock_memory_class.get_memory_instance.return_value = mock_memory

            response = client.get("/api/labels")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Should have both env values sorted
            assert set(data["labels"]["env"]) == {"prod", "staging"}
            assert data["labels"]["team"] == ["blue"]

    def test_get_labels_skips_internal_metadata(self, client: TestClient) -> None:
        """Test that internal metadata keys are skipped."""
        mock_ar = MagicMock()
        mock_ar.metadata = {
            "_internal": "value",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-02",
            "visible_label": "keep",
        }

        with patch("pyrit.backend.routes.labels.CentralMemory") as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory.get_attack_results.return_value = [mock_ar]
            mock_memory_class.get_memory_instance.return_value = mock_memory

            response = client.get("/api/labels")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Only visible_label should be included
            assert "visible_label" in data["labels"]
            assert "_internal" not in data["labels"]
            assert "created_at" not in data["labels"]
            assert "updated_at" not in data["labels"]

    def test_get_labels_skips_non_string_values(self, client: TestClient) -> None:
        """Test that non-string metadata values are skipped."""
        mock_ar = MagicMock()
        mock_ar.metadata = {
            "string_val": "keep",
            "int_val": 123,
            "list_val": ["a", "b"],
            "dict_val": {"nested": "value"},
        }

        with patch("pyrit.backend.routes.labels.CentralMemory") as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory.get_attack_results.return_value = [mock_ar]
            mock_memory_class.get_memory_instance.return_value = mock_memory

            response = client.get("/api/labels")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Only string_val should be included
            assert "string_val" in data["labels"]
            assert "int_val" not in data["labels"]
            assert "list_val" not in data["labels"]
            assert "dict_val" not in data["labels"]
