# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for attack service.

The attack service uses PyRIT memory with AttackResult as the source of truth.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.models.attacks import (
    AddMessageRequest,
    CreateAttackRequest,
    MessagePieceRequest,
    UpdateAttackRequest,
)
from pyrit.backend.services.attack_service import (
    AttackService,
    get_attack_service,
)
from pyrit.models import AttackOutcome, AttackResult


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    memory = MagicMock()
    memory.get_attack_results.return_value = []
    memory.get_conversation.return_value = []
    memory.get_message_pieces.return_value = []
    return memory


@pytest.fixture
def attack_service(mock_memory):
    """Create an attack service with mocked memory."""
    with patch("pyrit.backend.services.attack_service.CentralMemory") as mock_central:
        mock_central.get_memory_instance.return_value = mock_memory
        service = AttackService()
        yield service


def make_attack_result(
    *,
    conversation_id: str = "attack-1",
    objective: str = "Test objective",
    target_id: str = "target-1",
    target_type: str = "TextTarget",
    name: str = "Test Attack",
    outcome: AttackOutcome = AttackOutcome.UNDETERMINED,
    created_at: datetime = None,
    updated_at: datetime = None,
) -> AttackResult:
    """Create a mock AttackResult for testing."""
    now = datetime.now(timezone.utc)
    created = created_at or now
    updated = updated_at or now
    return AttackResult(
        conversation_id=conversation_id,
        objective=objective,
        attack_identifier={
            "name": name,
            "target_id": target_id,
            "target_type": target_type,
            "source": "gui",
        },
        outcome=outcome,
        metadata={
            "created_at": created.isoformat(),
            "updated_at": updated.isoformat(),
        },
    )


def make_mock_piece(
    *,
    conversation_id: str,
    role: str = "user",
    sequence: int = 0,
    original_value: str = "test",
    converted_value: str = "test",
    timestamp: datetime = None,
):
    """Create a mock message piece."""
    piece = MagicMock()
    piece.id = "piece-id"
    piece.conversation_id = conversation_id
    piece.role = role
    piece.sequence = sequence
    piece.original_value = original_value
    piece.converted_value = converted_value
    piece.converted_value_data_type = "text"
    piece.response_error = "none"
    piece.timestamp = timestamp or datetime.now(timezone.utc)
    piece.scores = []
    return piece


def make_mock_message(pieces: list):
    """Create a mock message from pieces."""
    msg = MagicMock()
    msg.message_pieces = pieces
    return msg


# ============================================================================
# Init Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAttackServiceInit:
    """Tests for AttackService initialization."""

    def test_init_gets_memory_instance(self) -> None:
        """Test that init gets the memory instance."""
        with patch("pyrit.backend.services.attack_service.CentralMemory") as mock_central:
            mock_memory = MagicMock()
            mock_central.get_memory_instance.return_value = mock_memory

            service = AttackService()

            mock_central.get_memory_instance.assert_called_once()
            assert service._memory == mock_memory


# ============================================================================
# List Attacks Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestListAttacks:
    """Tests for list_attacks method."""

    @pytest.mark.asyncio
    async def test_list_attacks_returns_empty_when_no_attacks(
        self, attack_service, mock_memory
    ) -> None:
        """Test that list_attacks returns empty list when no AttackResults exist."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.list_attacks()

        assert result.items == []
        assert result.pagination.has_more is False

    @pytest.mark.asyncio
    async def test_list_attacks_returns_attacks(
        self, attack_service, mock_memory
    ) -> None:
        """Test that list_attacks returns attacks from AttackResult records."""
        ar = make_attack_result()
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks()

        assert len(result.items) == 1
        assert result.items[0].attack_id == "attack-1"
        assert result.items[0].target_id == "target-1"

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_target_id(
        self, attack_service, mock_memory
    ) -> None:
        """Test that list_attacks filters by target_id."""
        ar1 = make_attack_result(conversation_id="attack-1", target_id="target-1")
        ar2 = make_attack_result(conversation_id="attack-2", target_id="target-2")
        mock_memory.get_attack_results.return_value = [ar1, ar2]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks(target_id="target-1")

        assert len(result.items) == 1
        assert result.items[0].target_id == "target-1"


# ============================================================================
# Get Attack Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestGetAttack:
    """Tests for get_attack method."""

    @pytest.mark.asyncio
    async def test_get_attack_returns_none_for_nonexistent(
        self, attack_service, mock_memory
    ) -> None:
        """Test that get_attack returns None when AttackResult doesn't exist."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.get_attack("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_attack_returns_attack_details(
        self, attack_service, mock_memory
    ) -> None:
        """Test that get_attack returns attack details from AttackResult."""
        ar = make_attack_result(
            conversation_id="test-id",
            name="My Attack",
            target_id="target-1",
            target_type="TextTarget",
        )
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        result = await attack_service.get_attack("test-id")

        assert result is not None
        assert result.attack_id == "test-id"
        assert result.target_id == "target-1"
        assert result.target_type == "TextTarget"
        assert result.name == "My Attack"


# ============================================================================
# Create Attack Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestCreateAttack:
    """Tests for create_attack method."""

    @pytest.mark.asyncio
    async def test_create_attack_validates_target_exists(self, attack_service) -> None:
        """Test that create_attack validates target exists."""
        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target = AsyncMock(return_value=None)
            mock_get_target_service.return_value = mock_target_service

            with pytest.raises(ValueError, match="not found"):
                await attack_service.create_attack(
                    CreateAttackRequest(target_id="nonexistent")
                )

    @pytest.mark.asyncio
    async def test_create_attack_stores_attack_result(
        self, attack_service, mock_memory
    ) -> None:
        """Test that create_attack stores AttackResult in memory."""
        with patch(
            "pyrit.backend.services.attack_service.get_target_service"
        ) as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target = AsyncMock(
                return_value=MagicMock(type="TextTarget")
            )
            mock_get_target_service.return_value = mock_target_service

            result = await attack_service.create_attack(
                CreateAttackRequest(target_id="target-1", name="My Attack")
            )

            assert result.attack_id is not None
            assert result.created_at is not None
            mock_memory.add_attack_results_to_memory.assert_called_once()


# ============================================================================
# Update Attack Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestUpdateAttack:
    """Tests for update_attack method."""

    @pytest.mark.asyncio
    async def test_update_attack_returns_none_for_nonexistent(
        self, attack_service, mock_memory
    ) -> None:
        """Test that update_attack returns None for nonexistent attack."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.update_attack(
            "nonexistent", UpdateAttackRequest(outcome="success")
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_update_attack_updates_outcome(
        self, attack_service, mock_memory
    ) -> None:
        """Test that update_attack updates the AttackResult outcome."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        await attack_service.update_attack(
            "test-id", UpdateAttackRequest(outcome="success")
        )

        # Should call add_attack_results_to_memory to update
        mock_memory.add_attack_results_to_memory.assert_called()


# ============================================================================
# Add Message Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAddMessage:
    """Tests for add_message method."""

    @pytest.mark.asyncio
    async def test_add_message_raises_for_nonexistent_attack(
        self, attack_service, mock_memory
    ) -> None:
        """Test that add_message raises ValueError for nonexistent attack."""
        mock_memory.get_attack_results.return_value = []

        request = AddMessageRequest(
            pieces=[MessagePieceRequest(original_value="Hello")],
        )

        with pytest.raises(ValueError, match="not found"):
            await attack_service.add_message("nonexistent", request)

    @pytest.mark.asyncio
    async def test_add_message_without_send_stores_message(
        self, attack_service, mock_memory
    ) -> None:
        """Test that add_message with send=False stores message in memory."""
        ar = make_attack_result(conversation_id="test-id", target_id="target-1")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []
        mock_memory.get_conversation.return_value = []

        request = AddMessageRequest(
            role="system",
            pieces=[MessagePieceRequest(original_value="You are a helpful assistant.")],
            send=False,
        )

        result = await attack_service.add_message("test-id", request)

        assert result.attack is not None
        mock_memory.add_message_pieces_to_memory.assert_called()


# ============================================================================
# Singleton Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAttackServiceSingleton:
    """Tests for get_attack_service singleton function."""

    def test_get_attack_service_returns_attack_service(self) -> None:
        """Test that get_attack_service returns an AttackService instance."""
        # Reset singleton for clean test
        import pyrit.backend.services.attack_service as module

        module._attack_service = None

        with patch("pyrit.backend.services.attack_service.CentralMemory"):
            service = get_attack_service()
            assert isinstance(service, AttackService)

    def test_get_attack_service_returns_same_instance(self) -> None:
        """Test that get_attack_service returns the same instance."""
        # Reset singleton for clean test
        import pyrit.backend.services.attack_service as module

        module._attack_service = None

        with patch("pyrit.backend.services.attack_service.CentralMemory"):
            service1 = get_attack_service()
            service2 = get_attack_service()
            assert service1 is service2
