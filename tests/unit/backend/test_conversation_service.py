# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend conversation service.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pyrit.backend.models.conversations import (
    CreateConversationRequest,
    ConverterConfig,
)
from pyrit.backend.services.conversation_service import (
    ConversationService,
    ConversationState,
    get_conversation_service,
)


class TestConversationState:
    """Tests for ConversationState model."""

    def test_conversation_state_creation(self) -> None:
        """Test creating a conversation state."""
        state = ConversationState(
            conversation_id="test-id",
            target_class="OpenAIChatTarget",
            target_identifier={"endpoint": "test"},
            created_at=datetime.utcnow(),
        )

        assert state.conversation_id == "test-id"
        assert state.target_class == "OpenAIChatTarget"
        assert state.converters == []

    def test_conversation_state_with_system_prompt(self) -> None:
        """Test conversation state with system prompt."""
        state = ConversationState(
            conversation_id="test-id",
            target_class="OpenAIChatTarget",
            target_identifier={},
            system_prompt="Test prompt",
            created_at=datetime.utcnow(),
        )

        assert state.system_prompt == "Test prompt"

    def test_conversation_state_defaults(self) -> None:
        """Test conversation state default values."""
        state = ConversationState(
            conversation_id="test-id",
            target_class="OpenAIChatTarget",
            target_identifier={},
            created_at=datetime.utcnow(),
        )

        assert state.system_prompt is None
        assert state.converters == []
        assert state.message_count == 0
        assert state.labels is None


class TestConversationService:
    """Tests for ConversationService."""

    @pytest.fixture
    def service(self, patch_central_database: MagicMock) -> ConversationService:
        """Create a conversation service instance.

        Args:
            patch_central_database: The patched central database fixture.

        Returns:
            ConversationService: The service instance.
        """
        return ConversationService()

    @pytest.mark.asyncio
    async def test_create_conversation_success(
        self, service: ConversationService
    ) -> None:
        """Test creating a conversation successfully."""
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"type": "TextTarget"}

        with patch.object(
            service, "_instantiate_target_from_class", return_value=mock_target
        ):
            request = CreateConversationRequest(target_class="TextTarget")
            result = await service.create_conversation(request)

            assert result is not None
            assert result.conversation_id is not None

    @pytest.mark.asyncio
    async def test_create_conversation_with_labels(
        self, service: ConversationService
    ) -> None:
        """Test creating a conversation with labels."""
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"type": "TextTarget"}

        with patch.object(
            service, "_instantiate_target_from_class", return_value=mock_target
        ):
            request = CreateConversationRequest(
                target_class="TextTarget",
                labels={"test": "label"},
            )
            result = await service.create_conversation(request)

            assert result.labels == {"test": "label"}

    @pytest.mark.asyncio
    async def test_create_conversation_invalid_target_class(
        self, service: ConversationService
    ) -> None:
        """Test creating a conversation with invalid target class."""
        with patch.object(
            service,
            "_instantiate_target_from_class",
            side_effect=ValueError("Target class 'InvalidTarget' not found"),
        ):
            request = CreateConversationRequest(target_class="InvalidTarget")

            with pytest.raises(ValueError, match="not found"):
                await service.create_conversation(request)

    @pytest.mark.asyncio
    async def test_get_conversation_existing(
        self, service: ConversationService
    ) -> None:
        """Test getting an existing conversation."""
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"type": "TextTarget"}

        with patch.object(
            service, "_instantiate_target_from_class", return_value=mock_target
        ):
            request = CreateConversationRequest(target_class="TextTarget")
            created = await service.create_conversation(request)

            result = await service.get_conversation(created.conversation_id)

            assert result is not None
            assert result.conversation_id == created.conversation_id

    @pytest.mark.asyncio
    async def test_get_conversation_nonexistent(
        self, service: ConversationService
    ) -> None:
        """Test getting a nonexistent conversation."""
        result = await service.get_conversation("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_conversation_messages_returns_list(
        self, service: ConversationService
    ) -> None:
        """Test getting messages from a conversation."""
        messages = await service.get_conversation_messages("any-id")

        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_cleanup_conversation_existing(
        self, service: ConversationService
    ) -> None:
        """Test cleaning up an existing conversation."""
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"type": "TextTarget"}

        with patch.object(
            service, "_instantiate_target_from_class", return_value=mock_target
        ):
            request = CreateConversationRequest(target_class="TextTarget")
            created = await service.create_conversation(request)

            service.cleanup_conversation(created.conversation_id)

            result = await service.get_conversation(created.conversation_id)
            assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_conversation_removes_target_instance(
        self, service: ConversationService
    ) -> None:
        """Test that cleanup removes target instance."""
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = {"type": "TextTarget"}

        with patch.object(
            service, "_instantiate_target_from_class", return_value=mock_target
        ):
            request = CreateConversationRequest(target_class="TextTarget")
            created = await service.create_conversation(request)

            assert created.conversation_id in service._target_instances

            service.cleanup_conversation(created.conversation_id)

            assert created.conversation_id not in service._target_instances

    def test_cleanup_conversation_nonexistent_no_error(
        self, service: ConversationService
    ) -> None:
        """Test cleaning up nonexistent conversation doesn't raise error."""
        # Should not raise any exception
        service.cleanup_conversation("nonexistent-id")


class TestGetConversationServiceSingleton:
    """Tests for get_conversation_service singleton function."""

    def test_returns_conversation_service_instance(
        self, patch_central_database: MagicMock
    ) -> None:
        """Test that get_conversation_service returns a ConversationService."""
        import pyrit.backend.services.conversation_service as module

        module._conversation_service = None

        service = get_conversation_service()

        assert isinstance(service, ConversationService)

    def test_returns_same_instance(
        self, patch_central_database: MagicMock
    ) -> None:
        """Test that get_conversation_service returns the same instance."""
        import pyrit.backend.services.conversation_service as module

        module._conversation_service = None

        service1 = get_conversation_service()
        service2 = get_conversation_service()

        assert service1 is service2
