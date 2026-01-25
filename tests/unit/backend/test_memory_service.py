# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend memory service.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pyrit.backend.models.common import PaginatedResponse
from pyrit.backend.services.memory_service import (
    MemoryService,
    get_memory_service,
    _parse_cursor,
    _build_cursor,
)


class TestCursorFunctions:
    """Tests for cursor parsing and building functions."""

    def test_parse_cursor_with_valid_cursor(self) -> None:
        """Test parsing a valid cursor string."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        cursor = f"{timestamp.isoformat()}_abc123"

        parsed_time, parsed_id = _parse_cursor(cursor)

        assert parsed_id == "abc123"
        assert parsed_time is not None
        assert parsed_time.year == 2024

    def test_parse_cursor_with_none(self) -> None:
        """Test parsing None cursor."""
        parsed_time, parsed_id = _parse_cursor(None)

        assert parsed_time is None
        assert parsed_id is None

    def test_parse_cursor_with_empty_string(self) -> None:
        """Test parsing empty cursor string."""
        parsed_time, parsed_id = _parse_cursor("")

        assert parsed_time is None
        assert parsed_id is None

    def test_parse_cursor_with_invalid_format(self) -> None:
        """Test parsing cursor with invalid format."""
        parsed_time, parsed_id = _parse_cursor("invalid_cursor_without_timestamp")

        assert parsed_time is None
        assert parsed_id is None

    def test_parse_cursor_with_malformed_timestamp(self) -> None:
        """Test parsing cursor with malformed timestamp."""
        parsed_time, parsed_id = _parse_cursor("not-a-timestamp_abc123")

        assert parsed_time is None
        assert parsed_id is None

    def test_build_cursor_creates_valid_string(self) -> None:
        """Test building a cursor string."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        record_id = "test-id-123"

        cursor = _build_cursor(timestamp, record_id)

        assert record_id in cursor
        assert timestamp.isoformat() in cursor

    def test_cursor_roundtrip(self) -> None:
        """Test that a cursor can be built and parsed back."""
        original_time = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        original_id = "message-uuid-123"

        cursor = _build_cursor(original_time, original_id)
        parsed_time, parsed_id = _parse_cursor(cursor)

        assert parsed_id == original_id
        assert parsed_time is not None


class TestMemoryService:
    """Tests for MemoryService."""

    @pytest.fixture
    def service(self, patch_central_database: MagicMock) -> MemoryService:
        """Create a memory service with patched database.

        Args:
            patch_central_database: The patched central database fixture.

        Returns:
            MemoryService: The service instance.
        """
        return MemoryService()

    @pytest.mark.asyncio
    async def test_get_messages_returns_paginated_result(
        self, service: MemoryService
    ) -> None:
        """Test that get_messages returns paginated results."""
        result = await service.get_messages()

        assert isinstance(result, PaginatedResponse)
        assert isinstance(result.items, list)
        assert result.pagination is not None

    @pytest.mark.asyncio
    async def test_get_messages_with_conversation_id(
        self, service: MemoryService
    ) -> None:
        """Test filtering messages by conversation ID."""
        result = await service.get_messages(conversation_id="test-conv-id")

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_messages_respects_limit(
        self, service: MemoryService
    ) -> None:
        """Test that limit parameter is respected."""
        result = await service.get_messages(limit=10)

        assert len(result.items) <= 10

    @pytest.mark.asyncio
    async def test_get_messages_with_role_filter(
        self, service: MemoryService
    ) -> None:
        """Test filtering messages by role."""
        result = await service.get_messages(role="user")

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_messages_with_time_filters(
        self, service: MemoryService
    ) -> None:
        """Test filtering messages by time range."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = await service.get_messages(start_time=start, end_time=end)

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_messages_pagination_has_more(
        self, service: MemoryService
    ) -> None:
        """Test that pagination correctly reports has_more."""
        result = await service.get_messages(limit=1)

        assert isinstance(result.pagination.has_more, bool)

    @pytest.mark.asyncio
    async def test_get_scores_returns_paginated_result(
        self, service: MemoryService
    ) -> None:
        """Test that get_scores returns paginated results."""
        result = await service.get_scores()

        assert isinstance(result, PaginatedResponse)
        assert isinstance(result.items, list)
        assert result.pagination is not None

    @pytest.mark.asyncio
    async def test_get_scores_with_message_id(
        self, service: MemoryService
    ) -> None:
        """Test filtering scores by message ID."""
        result = await service.get_scores(message_id="test-message-id")

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_scores_with_score_type(
        self, service: MemoryService
    ) -> None:
        """Test filtering scores by score type."""
        result = await service.get_scores(score_type="true_false")

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_attack_results_returns_paginated_result(
        self, service: MemoryService
    ) -> None:
        """Test that get_attack_results returns paginated results."""
        result = await service.get_attack_results()

        assert isinstance(result, PaginatedResponse)
        assert isinstance(result.items, list)
        assert result.pagination is not None

    @pytest.mark.asyncio
    async def test_get_attack_results_with_outcome_filter(
        self, service: MemoryService
    ) -> None:
        """Test filtering attack results by outcome."""
        result = await service.get_attack_results(outcome="success")

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_attack_results_with_turn_filters(
        self, service: MemoryService
    ) -> None:
        """Test filtering attack results by turn count."""
        result = await service.get_attack_results(min_turns=1, max_turns=10)

        assert isinstance(result, PaginatedResponse)

    @pytest.mark.asyncio
    async def test_get_seeds_returns_paginated_result(
        self, service: MemoryService
    ) -> None:
        """Test that get_seeds returns paginated results."""
        result = await service.get_seeds()

        assert isinstance(result, PaginatedResponse)
        assert isinstance(result.items, list)
        assert result.pagination is not None

    @pytest.mark.asyncio
    async def test_get_scenario_results_returns_paginated_result(
        self, service: MemoryService
    ) -> None:
        """Test that get_scenario_results returns paginated results."""
        result = await service.get_scenario_results()

        assert isinstance(result, PaginatedResponse)
        assert isinstance(result.items, list)
        assert result.pagination is not None


class TestGetMemoryServiceSingleton:
    """Tests for get_memory_service singleton function."""

    def test_returns_memory_service_instance(
        self, patch_central_database: MagicMock
    ) -> None:
        """Test that get_memory_service returns a MemoryService."""
        import pyrit.backend.services.memory_service as module

        module._memory_service = None

        service = get_memory_service()

        assert isinstance(service, MemoryService)

    def test_returns_same_instance(
        self, patch_central_database: MagicMock
    ) -> None:
        """Test that get_memory_service returns the same instance."""
        import pyrit.backend.services.memory_service as module

        module._memory_service = None

        service1 = get_memory_service()
        service2 = get_memory_service()

        assert service1 is service2
