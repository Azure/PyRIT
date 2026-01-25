# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend API routes.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pyrit.backend.models.common import PaginatedResponse, PaginationInfo
from pyrit.backend.models.conversations import CreateConversationResponse
from pyrit.backend.models.memory import MessageQueryResponse
from pyrit.backend.routes import health, version


class TestHealthRoute:
    """Tests for health check endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for health routes.

        Returns:
            TestClient: The test client.
        """
        app = FastAPI()
        app.include_router(health.router)
        return TestClient(app)

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test that health endpoint returns 200."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_returns_service_name(self, client: TestClient) -> None:
        """Test that health endpoint returns service name."""
        response = client.get("/health")
        data = response.json()

        assert data["service"] == "pyrit-backend"

    def test_health_returns_timestamp(self, client: TestClient) -> None:
        """Test that health endpoint returns timestamp."""
        response = client.get("/health")
        data = response.json()

        assert "timestamp" in data
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))


class TestVersionRoute:
    """Tests for version endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for version routes.

        Returns:
            TestClient: The test client.
        """
        app = FastAPI()
        app.include_router(version.router)
        return TestClient(app)

    def test_version_returns_200(self, client: TestClient) -> None:
        """Test that version endpoint returns 200."""
        response = client.get("/api/version")

        assert response.status_code == 200

    def test_version_returns_version_string(self, client: TestClient) -> None:
        """Test that version endpoint returns version string."""
        response = client.get("/api/version")
        data = response.json()

        assert "version" in data
        assert isinstance(data["version"], str)

    def test_version_returns_display_string(self, client: TestClient) -> None:
        """Test that version endpoint returns display string."""
        response = client.get("/api/version")
        data = response.json()

        assert "display" in data
        assert isinstance(data["display"], str)


class TestRegistryRoutes:
    """Tests for registry endpoints."""

    @pytest.fixture
    def mock_service(self) -> MagicMock:
        """Create a mock registry service.

        Returns:
            MagicMock: The mock service.
        """
        return MagicMock()

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for registry routes.

        Returns:
            TestClient: The test client.
        """
        from pyrit.backend.routes import registry

        app = FastAPI()
        app.include_router(registry.router)
        return TestClient(app)

    def test_list_targets_returns_200(self, client: TestClient) -> None:
        """Test that list targets returns 200."""
        mock_service = MagicMock()
        mock_service.get_targets.return_value = []

        with patch(
            "pyrit.backend.routes.registry.get_registry_service",
            return_value=mock_service,
        ):
            response = client.get("/registry/targets")

        assert response.status_code == 200

    def test_list_targets_returns_list(self, client: TestClient) -> None:
        """Test that list targets returns a list."""
        mock_service = MagicMock()
        mock_service.get_targets.return_value = []

        with patch(
            "pyrit.backend.routes.registry.get_registry_service",
            return_value=mock_service,
        ):
            response = client.get("/registry/targets")
            data = response.json()

        assert isinstance(data, list)

    def test_list_converters_returns_200(self, client: TestClient) -> None:
        """Test that list converters returns 200."""
        mock_service = MagicMock()
        mock_service.get_converters.return_value = []

        with patch(
            "pyrit.backend.routes.registry.get_registry_service",
            return_value=mock_service,
        ):
            response = client.get("/registry/converters")

        assert response.status_code == 200

    def test_list_scorers_returns_200(self, client: TestClient) -> None:
        """Test that list scorers returns 200."""
        mock_service = MagicMock()
        mock_service.get_scorers.return_value = []

        with patch(
            "pyrit.backend.routes.registry.get_registry_service",
            return_value=mock_service,
        ):
            response = client.get("/registry/scorers")

        assert response.status_code == 200

    def test_list_scenarios_returns_200(self, client: TestClient) -> None:
        """Test that list scenarios returns 200."""
        mock_service = MagicMock()
        mock_service.get_scenarios.return_value = []

        with patch(
            "pyrit.backend.routes.registry.get_registry_service",
            return_value=mock_service,
        ):
            response = client.get("/registry/scenarios")

        assert response.status_code == 200


class TestConversationRoutes:
    """Tests for conversation endpoints."""

    @pytest.fixture
    def client(self, patch_central_database: MagicMock) -> TestClient:
        """Create a test client for conversation routes.

        Args:
            patch_central_database: The patched central database fixture.

        Returns:
            TestClient: The test client.
        """
        from pyrit.backend.routes import conversations

        app = FastAPI()
        app.include_router(conversations.router)
        return TestClient(app)

    def test_create_conversation_returns_201(self, client: TestClient, patch_central_database: MagicMock) -> None:
        """Test that create conversation returns 201."""
        mock_service = MagicMock()
        mock_response = CreateConversationResponse(
            conversation_id="test-id",
            target_identifier={"__type__": "TextTarget"},
            labels=None,
            created_at=datetime.now(),
        )
        mock_service.create_conversation = AsyncMock(return_value=mock_response)

        with patch(
            "pyrit.backend.routes.conversations.get_conversation_service",
            return_value=mock_service,
        ):
            response = client.post(
                "/conversations",
                json={
                    "target_class": "TextTarget",
                    "target_params": None,
                },
            )

        assert response.status_code == 201

    def test_get_conversation_returns_404_for_missing(
        self, client: TestClient, patch_central_database: MagicMock
    ) -> None:
        """Test that get conversation returns 404 for missing."""
        mock_service = MagicMock()
        mock_service.get_conversation = AsyncMock(return_value=None)

        with patch(
            "pyrit.backend.routes.conversations.get_conversation_service",
            return_value=mock_service,
        ):
            response = client.get("/conversations/nonexistent")

        assert response.status_code == 404

    def test_delete_conversation_returns_204(self, client: TestClient, patch_central_database: MagicMock) -> None:
        """Test that delete conversation returns 204."""
        mock_service = MagicMock()
        # Must return a conversation state for delete to work
        mock_service.get_conversation = AsyncMock(return_value=MagicMock(conversation_id="conv-1"))
        mock_service.cleanup_conversation = MagicMock()

        with patch(
            "pyrit.backend.routes.conversations.get_conversation_service",
            return_value=mock_service,
        ):
            response = client.delete("/conversations/conv-1")

        assert response.status_code == 204


class TestMemoryRoutes:
    """Tests for memory endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for memory routes.

        Returns:
            TestClient: The test client.
        """
        from pyrit.backend.routes import memory

        app = FastAPI()
        app.include_router(memory.router)
        return TestClient(app)

    def test_query_messages_returns_200(self, client: TestClient) -> None:
        """Test that query messages returns 200."""
        mock_service = MagicMock()
        mock_response = PaginatedResponse[MessageQueryResponse](
            items=[],
            pagination=PaginationInfo(offset=0, limit=50, total=0, has_more=False),
        )
        mock_service.get_messages = AsyncMock(return_value=mock_response)

        with patch(
            "pyrit.backend.routes.memory.get_memory_service",
            return_value=mock_service,
        ):
            response = client.get("/memory/messages")

        assert response.status_code == 200

    def test_query_scores_returns_200(self, client: TestClient) -> None:
        """Test that query scores returns 200."""
        mock_service = MagicMock()
        mock_response = PaginatedResponse(
            items=[],
            pagination=PaginationInfo(offset=0, limit=50, total=0, has_more=False),
        )
        mock_service.get_scores = AsyncMock(return_value=mock_response)

        with patch(
            "pyrit.backend.routes.memory.get_memory_service",
            return_value=mock_service,
        ):
            response = client.get("/memory/scores")

        assert response.status_code == 200

    def test_query_attack_results_returns_200(self, client: TestClient) -> None:
        """Test that query attack results returns 200."""
        mock_service = MagicMock()
        mock_response = PaginatedResponse(
            items=[],
            pagination=PaginationInfo(offset=0, limit=50, total=0, has_more=False),
        )
        mock_service.get_attack_results = AsyncMock(return_value=mock_response)

        with patch(
            "pyrit.backend.routes.memory.get_memory_service",
            return_value=mock_service,
        ):
            response = client.get("/memory/attack-results")

        assert response.status_code == 200

    def test_query_seeds_returns_200(self, client: TestClient) -> None:
        """Test that query seeds returns 200."""
        mock_service = MagicMock()
        mock_response = PaginatedResponse(
            items=[],
            pagination=PaginationInfo(offset=0, limit=50, total=0, has_more=False),
        )
        mock_service.get_seeds = AsyncMock(return_value=mock_response)

        with patch(
            "pyrit.backend.routes.memory.get_memory_service",
            return_value=mock_service,
        ):
            response = client.get("/memory/seeds")

        assert response.status_code == 200
