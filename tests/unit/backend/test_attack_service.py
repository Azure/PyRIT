# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend attack service.
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.models.attacks import (
    AddMessageRequest,
    CreateAttackRequest,
    MessagePieceRequest,
    PrependedMessageRequest,
    UpdateAttackRequest,
)
from pyrit.backend.models.targets import TargetInstance
from pyrit.backend.services.attack_service import AttackService, AttackState


@pytest.mark.usefixtures("patch_central_database")
class TestAttackServiceInit:
    """Tests for AttackService initialization."""

    def test_init_creates_empty_attacks_dict(self) -> None:
        """Test that service initializes with empty attacks dictionary."""
        service = AttackService()
        assert service._attacks == {}

    def test_init_creates_empty_messages_dict(self) -> None:
        """Test that service initializes with empty messages dictionary."""
        service = AttackService()
        assert len(service._messages) == 0


@pytest.mark.usefixtures("patch_central_database")
class TestListAttacks:
    """Tests for AttackService.list_attacks method."""

    @pytest.mark.asyncio
    async def test_list_attacks_returns_empty_when_no_attacks(self) -> None:
        """Test that list_attacks returns empty list when no attacks exist."""
        service = AttackService()

        result = await service.list_attacks()

        assert result.items == []
        assert result.pagination.has_more is False

    @pytest.mark.asyncio
    async def test_list_attacks_returns_attacks(self) -> None:
        """Test that list_attacks returns existing attacks."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        # Add a test attack
        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            name="Test Attack",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        result = await service.list_attacks()

        assert len(result.items) == 1
        assert result.items[0].attack_id == "test-id"
        assert result.items[0].name == "Test Attack"

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_target_id(self) -> None:
        """Test that list_attacks filters by target_id."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["attack-1"] = AttackState(
            attack_id="attack-1",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )
        service._attacks["attack-2"] = AttackState(
            attack_id="attack-2",
            target_id="target-2",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        result = await service.list_attacks(target_id="target-1")

        assert len(result.items) == 1
        assert result.items[0].target_id == "target-1"

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_outcome(self) -> None:
        """Test that list_attacks filters by outcome."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["attack-1"] = AttackState(
            attack_id="attack-1",
            target_id="target-1",
            target_type="TextTarget",
            outcome="success",
            created_at=now,
            updated_at=now,
        )
        service._attacks["attack-2"] = AttackState(
            attack_id="attack-2",
            target_id="target-1",
            target_type="TextTarget",
            outcome="failure",
            created_at=now,
            updated_at=now,
        )

        result = await service.list_attacks(outcome="success")

        assert len(result.items) == 1
        assert result.items[0].outcome == "success"

    @pytest.mark.asyncio
    async def test_list_attacks_respects_limit(self) -> None:
        """Test that list_attacks respects the limit parameter."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        for i in range(5):
            service._attacks[f"attack-{i}"] = AttackState(
                attack_id=f"attack-{i}",
                target_id="target-1",
                target_type="TextTarget",
                created_at=now,
                updated_at=now,
            )

        result = await service.list_attacks(limit=2)

        assert len(result.items) == 2
        assert result.pagination.has_more is True

    @pytest.mark.asyncio
    async def test_list_attacks_cursor_pagination(self) -> None:
        """Test that list_attacks handles cursor-based pagination."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        # Create attacks with different updated_at times
        for i in range(3):
            service._attacks[f"attack-{i}"] = AttackState(
                attack_id=f"attack-{i}",
                target_id="target-1",
                target_type="TextTarget",
                created_at=now,
                updated_at=now,
            )

        # Get first page
        first_page = await service.list_attacks(limit=2)
        assert len(first_page.items) == 2

        # Get second page using cursor
        if first_page.pagination.next_cursor:
            second_page = await service.list_attacks(
                limit=2, cursor=first_page.pagination.next_cursor
            )
            assert len(second_page.items) == 1


@pytest.mark.usefixtures("patch_central_database")
class TestGetAttack:
    """Tests for AttackService.get_attack method."""

    @pytest.mark.asyncio
    async def test_get_attack_returns_none_for_nonexistent(self) -> None:
        """Test that get_attack returns None for non-existent attack."""
        service = AttackService()

        result = await service.get_attack("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_attack_returns_attack_details(self) -> None:
        """Test that get_attack returns full attack details."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            name="Test Attack",
            target_id="target-1",
            target_type="TextTarget",
            outcome="pending",
            created_at=now,
            updated_at=now,
        )

        result = await service.get_attack("test-id")

        assert result is not None
        assert result.attack_id == "test-id"
        assert result.name == "Test Attack"
        assert result.target_type == "TextTarget"


@pytest.mark.usefixtures("patch_central_database")
class TestCreateAttack:
    """Tests for AttackService.create_attack method."""

    @pytest.mark.asyncio
    async def test_create_attack_validates_target_exists(self) -> None:
        """Test that create_attack validates the target exists."""
        service = AttackService()

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target = AsyncMock(return_value=None)
            mock_get_target_service.return_value = mock_target_service

            request = CreateAttackRequest(target_id="nonexistent")

            with pytest.raises(ValueError, match="not found"):
                await service.create_attack(request)

    @pytest.mark.asyncio
    async def test_create_attack_success(self) -> None:
        """Test successful attack creation."""
        service = AttackService()

        mock_target = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            params={},
            created_at=datetime.now(timezone.utc),
            source="user",
        )

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target = AsyncMock(return_value=mock_target)
            mock_get_target_service.return_value = mock_target_service

            request = CreateAttackRequest(target_id="target-1", name="My Attack")

            result = await service.create_attack(request)

            assert result.attack_id is not None
            assert result.created_at is not None

            # Verify the attack was stored correctly by fetching it
            attack = await service.get_attack(result.attack_id)
            assert attack is not None
            assert attack.name == "My Attack"
            assert attack.target_id == "target-1"
            assert attack.target_type == "TextTarget"

    @pytest.mark.asyncio
    async def test_create_attack_with_prepended_conversation(self) -> None:
        """Test attack creation with prepended conversation."""
        service = AttackService()

        mock_target = TargetInstance(
            target_id="target-1",
            type="TextTarget",
            params={},
            created_at=datetime.now(timezone.utc),
            source="user",
        )

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target = AsyncMock(return_value=mock_target)
            mock_get_target_service.return_value = mock_target_service

            request = CreateAttackRequest(
                target_id="target-1",
                prepended_conversation=[
                    PrependedMessageRequest(role="system", content="You are a helpful assistant."),
                ],
            )

            result = await service.create_attack(request)

            # Verify the attack was stored with prepended conversation
            attack = await service.get_attack(result.attack_id)
            assert attack is not None
            assert len(attack.prepended_conversation) == 1
            assert attack.prepended_conversation[0].role == "system"


@pytest.mark.usefixtures("patch_central_database")
class TestUpdateAttack:
    """Tests for AttackService.update_attack method."""

    @pytest.mark.asyncio
    async def test_update_attack_returns_none_for_nonexistent(self) -> None:
        """Test that update_attack returns None for non-existent attack."""
        service = AttackService()

        request = UpdateAttackRequest(outcome="success")
        result = await service.update_attack("nonexistent", request)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_attack_updates_outcome(self) -> None:
        """Test that update_attack updates the outcome."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            outcome=None,
            created_at=now,
            updated_at=now,
        )

        request = UpdateAttackRequest(outcome="success")
        result = await service.update_attack("test-id", request)

        assert result is not None
        assert result.outcome == "success"


@pytest.mark.usefixtures("patch_central_database")
class TestDeleteAttack:
    """Tests for AttackService.delete_attack method."""

    @pytest.mark.asyncio
    async def test_delete_attack_returns_false_for_nonexistent(self) -> None:
        """Test that delete_attack returns False for non-existent attack."""
        service = AttackService()

        result = await service.delete_attack("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_attack_deletes_attack(self) -> None:
        """Test that delete_attack removes the attack."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        result = await service.delete_attack("test-id")

        assert result is True
        assert "test-id" not in service._attacks

    @pytest.mark.asyncio
    async def test_delete_attack_removes_messages(self) -> None:
        """Test that delete_attack also removes associated messages."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )
        service._messages["test-id"] = []

        await service.delete_attack("test-id")

        assert "test-id" not in service._messages


@pytest.mark.usefixtures("patch_central_database")
class TestAddMessage:
    """Tests for AttackService.add_message method."""

    @pytest.mark.asyncio
    async def test_add_message_raises_for_nonexistent_attack(self) -> None:
        """Test that add_message raises ValueError for non-existent attack."""
        service = AttackService()

        request = AddMessageRequest(
            pieces=[MessagePieceRequest(content="Hello")],
        )

        with pytest.raises(ValueError, match="Attack"):
            await service.add_message("nonexistent", request)

    @pytest.mark.asyncio
    async def test_add_message_with_send_raises_for_missing_target_object(self) -> None:
        """Test that add_message with send=True raises when target object is not found."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target_object.return_value = None
            mock_get_target_service.return_value = mock_target_service

            with patch(
                "pyrit.backend.services.attack_service.get_converter_service"
            ) as mock_get_converter_service:
                mock_converter_service = MagicMock()
                mock_get_converter_service.return_value = mock_converter_service

                request = AddMessageRequest(
                    pieces=[MessagePieceRequest(content="Hello")],
                    send=True,
                )

                with pytest.raises(ValueError, match="Target object"):
                    await service.add_message("test-id", request)

    @pytest.mark.asyncio
    async def test_add_message_without_send_stores_message(self) -> None:
        """Test that add_message with send=False just stores the message."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_get_target_service.return_value = mock_target_service

            with patch(
                "pyrit.backend.services.attack_service.get_converter_service"
            ) as mock_get_converter_service:
                mock_converter_service = MagicMock()
                mock_get_converter_service.return_value = mock_converter_service

                request = AddMessageRequest(
                    role="system",
                    pieces=[MessagePieceRequest(content="You are a helpful assistant.")],
                    send=False,
                )

                result = await service.add_message("test-id", request)

                assert result.attack is not None
                assert len(result.attack.messages) == 1
                assert result.attack.messages[0].role == "system"
                assert result.error is None

    @pytest.mark.asyncio
    async def test_add_message_with_converter_ids_applies_converters(self) -> None:
        """Test that add_message with converter_ids applies the converters."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        # Create mock converter
        mock_converter = MagicMock()
        mock_converter.convert_async = AsyncMock(
            return_value=MagicMock(output_text="converted text")
        )

        # Create mock target
        mock_target = AsyncMock()
        mock_target.send_prompt_async = AsyncMock(
            return_value=MagicMock(
                request_pieces=[
                    MagicMock(
                        original_value="assistant response",
                        converted_value="assistant response",
                        original_value_data_type="text",
                    )
                ],
                response_error_description="none",
            )
        )

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target_object.return_value = mock_target
            mock_get_target_service.return_value = mock_target_service

            with patch(
                "pyrit.backend.services.attack_service.get_converter_service"
            ) as mock_get_converter_service:
                mock_converter_service = MagicMock()
                mock_converter_service.get_converter_objects_for_ids.return_value = [
                    mock_converter
                ]
                mock_get_converter_service.return_value = mock_converter_service

                request = AddMessageRequest(
                    role="user",
                    pieces=[MessagePieceRequest(content="Hello")],
                    send=True,
                    converter_ids=["converter-1"],
                )

                result = await service.add_message("test-id", request)

                # Verify converter was applied
                mock_converter_service.get_converter_objects_for_ids.assert_called_once_with(
                    ["converter-1"]
                )
                mock_converter.convert_async.assert_called_once_with(prompt="Hello")

                # Verify message was converted
                assert result.attack is not None
                # First message is the user message with conversion
                user_msg = result.attack.messages[0]
                assert user_msg.role == "user"
                assert user_msg.pieces[0].converted_value == "converted text"

    @pytest.mark.asyncio
    async def test_add_message_without_converter_ids_does_not_apply_converters(self) -> None:
        """Test that add_message without converter_ids does not apply any converters."""
        service = AttackService()
        now = datetime.now(timezone.utc)

        service._attacks["test-id"] = AttackState(
            attack_id="test-id",
            target_id="target-1",
            target_type="TextTarget",
            created_at=now,
            updated_at=now,
        )

        # Create mock target
        mock_target = AsyncMock()
        mock_target.send_prompt_async = AsyncMock(
            return_value=MagicMock(
                request_pieces=[
                    MagicMock(
                        original_value="response",
                        converted_value="response",
                        original_value_data_type="text",
                    )
                ],
                response_error_description="none",
            )
        )

        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target_object.return_value = mock_target
            mock_get_target_service.return_value = mock_target_service

            with patch(
                "pyrit.backend.services.attack_service.get_converter_service"
            ) as mock_get_converter_service:
                mock_converter_service = MagicMock()
                mock_get_converter_service.return_value = mock_converter_service

                # No converter_ids in request - should not apply any converters
                request = AddMessageRequest(
                    role="user",
                    pieces=[MessagePieceRequest(content="Hello")],
                    send=True,
                )

                result = await service.add_message("test-id", request)

                # Verify no converter lookup was done
                mock_converter_service.get_converter_objects_for_ids.assert_not_called()

                # Verify original value equals converted value (no conversion)
                assert result.attack is not None
                user_msg = result.attack.messages[0]
                assert user_msg.pieces[0].original_value == "Hello"
                assert user_msg.pieces[0].converted_value == "Hello"


@pytest.mark.usefixtures("patch_central_database")
class TestAttackServiceSingleton:
    """Tests for get_attack_service singleton function."""

    def test_get_attack_service_returns_attack_service(self) -> None:
        """Test that get_attack_service returns an AttackService instance."""
        from pyrit.backend.services.attack_service import get_attack_service

        # Reset singleton for clean test
        import pyrit.backend.services.attack_service as module
        module._attack_service = None

        service = get_attack_service()
        assert isinstance(service, AttackService)

    def test_get_attack_service_returns_same_instance(self) -> None:
        """Test that get_attack_service returns the same instance."""
        from pyrit.backend.services.attack_service import get_attack_service

        # Reset singleton for clean test
        import pyrit.backend.services.attack_service as module
        module._attack_service = None

        service1 = get_attack_service()
        service2 = get_attack_service()
        assert service1 is service2
