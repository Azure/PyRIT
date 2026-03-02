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
    PrependedMessageRequest,
    UpdateAttackRequest,
)
from pyrit.backend.services.attack_service import (
    AttackService,
    get_attack_service,
)
from pyrit.identifiers import ComponentIdentifier
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
    has_target: bool = True,
    name: str = "Test Attack",
    outcome: AttackOutcome = AttackOutcome.UNDETERMINED,
    created_at: datetime = None,
    updated_at: datetime = None,
) -> AttackResult:
    """Create a mock AttackResult for testing."""
    now = datetime.now(timezone.utc)
    created = created_at or now
    updated = updated_at or now

    target_identifier = (
        ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        if has_target
        else None
    )

    return AttackResult(
        conversation_id=conversation_id,
        objective=objective,
        attack_identifier=ComponentIdentifier(
            class_name=name,
            class_module="pyrit.backend",
            children={"objective_target": target_identifier} if target_identifier else {},
        ),
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
    piece.get_role_for_storage.return_value = role
    piece.sequence = sequence
    piece.original_value = original_value
    piece.converted_value = converted_value
    piece.converted_value_data_type = "text"
    piece.original_value_data_type = "text"
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
    async def test_list_attacks_returns_empty_when_no_attacks(self, attack_service, mock_memory) -> None:
        """Test that list_attacks returns empty list when no AttackResults exist."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.list_attacks_async()

        assert result.items == []
        assert result.pagination.has_more is False

    @pytest.mark.asyncio
    async def test_list_attacks_returns_attacks(self, attack_service, mock_memory) -> None:
        """Test that list_attacks returns attacks from AttackResult records."""
        ar = make_attack_result()
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async()

        assert len(result.items) == 1
        assert result.items[0].conversation_id == "attack-1"
        assert result.items[0].attack_type == "Test Attack"

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_attack_class_exact(self, attack_service, mock_memory) -> None:
        """Test that list_attacks passes attack_class to memory layer."""
        ar1 = make_attack_result(conversation_id="attack-1", name="CrescendoAttack")
        mock_memory.get_attack_results.return_value = [ar1]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(attack_class="CrescendoAttack")

        assert len(result.items) == 1
        assert result.items[0].conversation_id == "attack-1"
        # Verify attack_class was forwarded to the memory layer
        call_kwargs = mock_memory.get_attack_results.call_args[1]
        assert call_kwargs["attack_class"] == "CrescendoAttack"

    @pytest.mark.asyncio
    async def test_list_attacks_attack_class_passed_to_memory(self, attack_service, mock_memory) -> None:
        """Test that attack_class is forwarded to memory for DB-level filtering."""
        mock_memory.get_attack_results.return_value = []
        mock_memory.get_message_pieces.return_value = []

        await attack_service.list_attacks_async(attack_class="Crescendo")

        call_kwargs = mock_memory.get_attack_results.call_args[1]
        assert call_kwargs["attack_class"] == "Crescendo"

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_no_converters(self, attack_service, mock_memory) -> None:
        """Test that converter_classes=[] is forwarded to memory for DB-level filtering."""
        mock_memory.get_attack_results.return_value = []
        mock_memory.get_message_pieces.return_value = []

        await attack_service.list_attacks_async(converter_classes=[])

        call_kwargs = mock_memory.get_attack_results.call_args[1]
        assert call_kwargs["converter_classes"] == []

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_converter_classes_and_logic(self, attack_service, mock_memory) -> None:
        """Test that list_attacks passes converter_classes to memory layer."""
        ar1 = make_attack_result(conversation_id="attack-1", name="Attack One")
        ar1.attack_identifier = ComponentIdentifier(
            class_name="Attack One",
            class_module="pyrit.backend",
            children={
                "request_converters": [
                    ComponentIdentifier(
                        class_name="Base64Converter",
                        class_module="pyrit.converters",
                        params={
                            "supported_input_types": ("text",),
                            "supported_output_types": ("text",),
                        },
                    ),
                    ComponentIdentifier(
                        class_name="ROT13Converter",
                        class_module="pyrit.converters",
                        params={
                            "supported_input_types": ("text",),
                            "supported_output_types": ("text",),
                        },
                    ),
                ],
            },
        )
        mock_memory.get_attack_results.return_value = [ar1]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(converter_classes=["Base64Converter", "ROT13Converter"])

        assert len(result.items) == 1
        assert result.items[0].conversation_id == "attack-1"
        # Verify converter_classes was forwarded to the memory layer
        call_kwargs = mock_memory.get_attack_results.call_args[1]
        assert call_kwargs["converter_classes"] == ["Base64Converter", "ROT13Converter"]

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_min_turns(self, attack_service, mock_memory) -> None:
        """Test that list_attacks filters by minimum executed turns."""
        ar1 = make_attack_result(conversation_id="attack-1")
        ar1.executed_turns = 5
        ar2 = make_attack_result(conversation_id="attack-2")
        ar2.executed_turns = 2
        mock_memory.get_attack_results.return_value = [ar1, ar2]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(min_turns=3)

        assert len(result.items) == 1
        assert result.items[0].conversation_id == "attack-1"

    @pytest.mark.asyncio
    async def test_list_attacks_filters_by_max_turns(self, attack_service, mock_memory) -> None:
        """Test that list_attacks filters by maximum executed turns."""
        ar1 = make_attack_result(conversation_id="attack-1")
        ar1.executed_turns = 5
        ar2 = make_attack_result(conversation_id="attack-2")
        ar2.executed_turns = 2
        mock_memory.get_attack_results.return_value = [ar1, ar2]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(max_turns=3)

        assert len(result.items) == 1
        assert result.items[0].conversation_id == "attack-2"

    @pytest.mark.asyncio
    async def test_list_attacks_includes_labels_in_summary(self, attack_service, mock_memory) -> None:
        """Test that list_attacks includes labels from message pieces in summaries."""
        ar = make_attack_result(
            conversation_id="attack-1",
        )
        mock_memory.get_attack_results.return_value = [ar]
        piece = make_mock_piece(conversation_id="attack-1")
        piece.labels = {"env": "prod", "team": "red"}
        mock_memory.get_message_pieces.return_value = [piece]

        result = await attack_service.list_attacks_async()

        assert len(result.items) == 1
        assert result.items[0].labels == {"env": "prod", "team": "red"}

    @pytest.mark.asyncio
    async def test_list_attacks_combined_min_and_max_turns(self, attack_service, mock_memory) -> None:
        """Test that list_attacks filters by both min_turns and max_turns together."""
        ar1 = make_attack_result(conversation_id="attack-1")
        ar1.executed_turns = 1
        ar2 = make_attack_result(conversation_id="attack-2")
        ar2.executed_turns = 3
        ar3 = make_attack_result(conversation_id="attack-3")
        ar3.executed_turns = 7
        mock_memory.get_attack_results.return_value = [ar1, ar2, ar3]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(min_turns=2, max_turns=5)

        assert len(result.items) == 1
        assert result.items[0].conversation_id == "attack-2"


# ============================================================================
# Attack Options Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAttackOptions:
    """Tests for get_attack_options_async method."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_attacks(self, attack_service, mock_memory) -> None:
        """Test that attack options returns empty list when no attacks exist."""
        mock_memory.get_unique_attack_class_names.return_value = []

        result = await attack_service.get_attack_options_async()

        assert result == []
        mock_memory.get_unique_attack_class_names.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_result_from_memory(self, attack_service, mock_memory) -> None:
        """Test that attack options delegates to memory layer."""
        mock_memory.get_unique_attack_class_names.return_value = ["CrescendoAttack", "ManualAttack"]

        result = await attack_service.get_attack_options_async()

        assert result == ["CrescendoAttack", "ManualAttack"]
        mock_memory.get_unique_attack_class_names.assert_called_once()


# ============================================================================
# Converter Options Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestConverterOptions:
    """Tests for get_converter_options_async method."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_attacks(self, attack_service, mock_memory) -> None:
        """Test that converter options returns empty list when no attacks exist."""
        mock_memory.get_unique_converter_class_names.return_value = []

        result = await attack_service.get_converter_options_async()

        assert result == []
        mock_memory.get_unique_converter_class_names.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_result_from_memory(self, attack_service, mock_memory) -> None:
        """Test that converter options delegates to memory layer."""
        mock_memory.get_unique_converter_class_names.return_value = ["Base64Converter", "ROT13Converter"]

        result = await attack_service.get_converter_options_async()

        assert result == ["Base64Converter", "ROT13Converter"]
        mock_memory.get_unique_converter_class_names.assert_called_once()


# ============================================================================
# Get Attack Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestGetAttack:
    """Tests for get_attack method."""

    @pytest.mark.asyncio
    async def test_get_attack_returns_none_for_nonexistent(self, attack_service, mock_memory) -> None:
        """Test that get_attack returns None when AttackResult doesn't exist."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.get_attack_async(conversation_id="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_attack_returns_attack_details(self, attack_service, mock_memory) -> None:
        """Test that get_attack returns attack details from AttackResult."""
        ar = make_attack_result(
            conversation_id="test-id",
            name="My Attack",
        )
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        result = await attack_service.get_attack_async(conversation_id="test-id")

        assert result is not None
        assert result.conversation_id == "test-id"
        assert result.attack_type == "My Attack"


# ============================================================================
# Get Attack Messages Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestGetAttackMessages:
    """Tests for get_attack_messages method."""

    @pytest.mark.asyncio
    async def test_get_attack_messages_returns_none_for_nonexistent(self, attack_service, mock_memory) -> None:
        """Test that get_attack_messages returns None when attack doesn't exist."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.get_attack_messages_async(conversation_id="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_attack_messages_returns_messages(self, attack_service, mock_memory) -> None:
        """Test that get_attack_messages returns messages for existing attack."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        result = await attack_service.get_attack_messages_async(conversation_id="test-id")

        assert result is not None
        assert result.conversation_id == "test-id"
        assert result.messages == []


# ============================================================================
# Create Attack Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestCreateAttack:
    """Tests for create_attack method."""

    @pytest.mark.asyncio
    async def test_create_attack_validates_target_exists(self, attack_service) -> None:
        """Test that create_attack validates target exists."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=None)
            mock_get_target_service.return_value = mock_target_service

            with pytest.raises(ValueError, match="not found"):
                await attack_service.create_attack_async(request=CreateAttackRequest(target_unique_name="nonexistent"))

    @pytest.mark.asyncio
    async def test_create_attack_stores_attack_result(self, attack_service, mock_memory) -> None:
        """Test that create_attack stores AttackResult in memory."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            result = await attack_service.create_attack_async(
                request=CreateAttackRequest(target_unique_name="target-1", name="My Attack")
            )

            assert result.conversation_id is not None
            assert result.created_at is not None
            mock_memory.add_attack_results_to_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_attack_stores_prepended_conversation(self, attack_service, mock_memory) -> None:
        """Test that create_attack stores prepended conversation messages."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            prepended = [
                PrependedMessageRequest(
                    role="system",
                    pieces=[MessagePieceRequest(original_value="You are a helpful assistant.")],
                )
            ]

            result = await attack_service.create_attack_async(
                request=CreateAttackRequest(target_unique_name="target-1", prepended_conversation=prepended)
            )

            assert result.conversation_id is not None
            # Both attack result and prepended message pieces should be stored
            mock_memory.add_attack_results_to_memory.assert_called_once()
            mock_memory.add_message_pieces_to_memory.assert_called()

    @pytest.mark.asyncio
    async def test_create_attack_does_not_store_labels_in_metadata(self, attack_service, mock_memory) -> None:
        """Test that labels are not stored in attack metadata (they live on pieces)."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            await attack_service.create_attack_async(
                request=CreateAttackRequest(
                    target_unique_name="target-1",
                    name="Labeled Attack",
                    labels={"env": "prod", "team": "red"},
                )
            )

            call_args = mock_memory.add_attack_results_to_memory.call_args
            stored_ar = call_args[1]["attack_results"][0]
            assert "labels" not in stored_ar.metadata

    @pytest.mark.asyncio
    async def test_create_attack_stamps_labels_on_prepended_pieces(self, attack_service, mock_memory) -> None:
        """Test that labels are forwarded to prepended message pieces."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            prepended = [
                PrependedMessageRequest(
                    role="system",
                    pieces=[MessagePieceRequest(original_value="Be helpful.")],
                )
            ]

            await attack_service.create_attack_async(
                request=CreateAttackRequest(
                    target_unique_name="target-1",
                    labels={"env": "prod"},
                    prepended_conversation=prepended,
                )
            )

            stored_piece = mock_memory.add_message_pieces_to_memory.call_args[1]["message_pieces"][0]
            assert stored_piece.labels == {"env": "prod", "source": "gui"}

    @pytest.mark.asyncio
    async def test_create_attack_prepended_messages_have_incrementing_sequences(
        self, attack_service, mock_memory
    ) -> None:
        """Test that multiple prepended messages get incrementing sequence numbers and preserve lineage."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            original_id_1 = "aaaaaaaa-1111-2222-3333-444444444444"
            original_id_2 = "bbbbbbbb-1111-2222-3333-444444444444"
            original_id_3 = "cccccccc-1111-2222-3333-444444444444"

            prepended = [
                PrependedMessageRequest(
                    role="system",
                    pieces=[
                        MessagePieceRequest(
                            original_value="You are a helpful assistant.",
                            original_prompt_id=original_id_1,
                        )
                    ],
                ),
                PrependedMessageRequest(
                    role="user",
                    pieces=[
                        MessagePieceRequest(original_value="Hello", original_prompt_id=original_id_2),
                    ],
                ),
                PrependedMessageRequest(
                    role="assistant",
                    pieces=[
                        MessagePieceRequest(original_value="Hi there!", original_prompt_id=original_id_3),
                    ],
                ),
            ]

            await attack_service.create_attack_async(
                request=CreateAttackRequest(target_unique_name="target-1", prepended_conversation=prepended)
            )

            # Each message stored separately with incrementing sequence
            calls = mock_memory.add_message_pieces_to_memory.call_args_list
            assert len(calls) == 3
            sequences = [call[1]["message_pieces"][0].sequence for call in calls]
            assert sequences == [0, 1, 2]

            roles = [call[1]["message_pieces"][0].api_role for call in calls]
            assert roles == ["system", "user", "assistant"]

            # original_prompt_id preserved for lineage tracking
            import uuid

            stored_pieces = [call[1]["message_pieces"][0] for call in calls]
            assert stored_pieces[0].original_prompt_id == uuid.UUID(original_id_1)
            assert stored_pieces[1].original_prompt_id == uuid.UUID(original_id_2)
            assert stored_pieces[2].original_prompt_id == uuid.UUID(original_id_3)

            # Each piece gets its own new id, different from the original
            for piece in stored_pieces:
                assert piece.id != piece.original_prompt_id

    @pytest.mark.asyncio
    async def test_create_attack_preserves_user_supplied_source_label(self, attack_service, mock_memory) -> None:
        """Test that setdefault does not overwrite user-supplied 'source' label."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            prepended = [
                PrependedMessageRequest(
                    role="system",
                    pieces=[MessagePieceRequest(original_value="Be helpful.")],
                )
            ]

            await attack_service.create_attack_async(
                request=CreateAttackRequest(
                    target_unique_name="target-1",
                    labels={"source": "api-test"},
                    prepended_conversation=prepended,
                )
            )

            stored_piece = mock_memory.add_message_pieces_to_memory.call_args[1]["message_pieces"][0]
            assert stored_piece.labels["source"] == "api-test"  # not overwritten to "gui"

    @pytest.mark.asyncio
    async def test_create_attack_default_name(self, attack_service, mock_memory) -> None:
        """Test that request.name=None uses default class_name and objective."""
        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_service:
            mock_target_obj = MagicMock()
            mock_target_obj.get_identifier.return_value = ComponentIdentifier(
                class_name="TextTarget", class_module="pyrit.prompt_target"
            )
            mock_target_service = MagicMock()
            mock_target_service.get_target_async = AsyncMock(return_value=MagicMock(type="TextTarget"))
            mock_target_service.get_target_object.return_value = mock_target_obj
            mock_get_target_service.return_value = mock_target_service

            await attack_service.create_attack_async(request=CreateAttackRequest(target_unique_name="target-1"))

            call_args = mock_memory.add_attack_results_to_memory.call_args
            stored_ar = call_args[1]["attack_results"][0]
            assert stored_ar.objective == "Manual attack via GUI"
            assert stored_ar.attack_identifier.class_name == "ManualAttack"


# ============================================================================
# Update Attack Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestUpdateAttack:
    """Tests for update_attack method."""

    @pytest.mark.asyncio
    async def test_update_attack_returns_none_for_nonexistent(self, attack_service, mock_memory) -> None:
        """Test that update_attack returns None for nonexistent attack."""
        mock_memory.get_attack_results.return_value = []

        result = await attack_service.update_attack_async(
            conversation_id="nonexistent", request=UpdateAttackRequest(outcome="success")
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_update_attack_updates_outcome_success(self, attack_service, mock_memory) -> None:
        """Test that update_attack maps 'success' to AttackOutcome.SUCCESS."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        await attack_service.update_attack_async(
            conversation_id="test-id", request=UpdateAttackRequest(outcome="success")
        )

        stored_ar = mock_memory.add_attack_results_to_memory.call_args[1]["attack_results"][0]
        assert stored_ar.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_update_attack_updates_outcome_failure(self, attack_service, mock_memory) -> None:
        """Test that update_attack maps 'failure' to AttackOutcome.FAILURE."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        await attack_service.update_attack_async(
            conversation_id="test-id", request=UpdateAttackRequest(outcome="failure")
        )

        stored_ar = mock_memory.add_attack_results_to_memory.call_args[1]["attack_results"][0]
        assert stored_ar.outcome == AttackOutcome.FAILURE

    @pytest.mark.asyncio
    async def test_update_attack_updates_outcome_undetermined(self, attack_service, mock_memory) -> None:
        """Test that update_attack maps 'undetermined' to AttackOutcome.UNDETERMINED."""
        ar = make_attack_result(conversation_id="test-id", outcome=AttackOutcome.SUCCESS)
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        await attack_service.update_attack_async(
            conversation_id="test-id", request=UpdateAttackRequest(outcome="undetermined")
        )

        stored_ar = mock_memory.add_attack_results_to_memory.call_args[1]["attack_results"][0]
        assert stored_ar.outcome == AttackOutcome.UNDETERMINED

    @pytest.mark.asyncio
    async def test_update_attack_refreshes_updated_at(self, attack_service, mock_memory) -> None:
        """Test that update_attack refreshes the updated_at metadata."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ar = make_attack_result(conversation_id="test-id", updated_at=old_time)
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_conversation.return_value = []

        await attack_service.update_attack_async(
            conversation_id="test-id", request=UpdateAttackRequest(outcome="success")
        )

        stored_ar = mock_memory.add_attack_results_to_memory.call_args[1]["attack_results"][0]
        assert stored_ar.metadata["updated_at"] != old_time.isoformat()


# ============================================================================
# Add Message Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAddMessage:
    """Tests for add_message method."""

    @pytest.mark.asyncio
    async def test_add_message_raises_for_nonexistent_attack(self, attack_service, mock_memory) -> None:
        """Test that add_message raises ValueError for nonexistent attack."""
        mock_memory.get_attack_results.return_value = []

        request = AddMessageRequest(
            pieces=[MessagePieceRequest(original_value="Hello")],
        )

        with pytest.raises(ValueError, match="not found"):
            await attack_service.add_message_async(conversation_id="nonexistent", request=request)

    @pytest.mark.asyncio
    async def test_add_message_without_send_stamps_labels_on_pieces(self, attack_service, mock_memory) -> None:
        """Test that add_message (send=False) inherits labels from existing pieces."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]

        existing_piece = make_mock_piece(conversation_id="test-id")
        existing_piece.labels = {"env": "prod"}
        mock_memory.get_message_pieces.return_value = [existing_piece]
        mock_memory.get_conversation.return_value = []

        request = AddMessageRequest(
            role="user",
            pieces=[MessagePieceRequest(original_value="Hello")],
            send=False,
        )

        result = await attack_service.add_message_async(conversation_id="test-id", request=request)

        stored_piece = mock_memory.add_message_pieces_to_memory.call_args[1]["message_pieces"][0]
        assert stored_piece.labels == {"env": "prod"}
        assert result.attack is not None

    @pytest.mark.asyncio
    async def test_add_message_with_send_passes_labels_to_normalizer(self, attack_service, mock_memory) -> None:
        """Test that add_message (send=True) inherits labels from existing pieces."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]

        existing_piece = make_mock_piece(conversation_id="test-id")
        existing_piece.labels = {"env": "staging"}
        mock_memory.get_message_pieces.return_value = [existing_piece]
        mock_memory.get_conversation.return_value = []

        with (
            patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_svc,
            patch("pyrit.backend.services.attack_service.PromptNormalizer") as mock_normalizer_cls,
        ):
            mock_target_svc = MagicMock()
            mock_target_svc.get_target_object.return_value = MagicMock()
            mock_get_target_svc.return_value = mock_target_svc

            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_cls.return_value = mock_normalizer

            request = AddMessageRequest(
                pieces=[MessagePieceRequest(original_value="Hello")],
                send=True,
            )

            await attack_service.add_message_async(conversation_id="test-id", request=request)

            call_kwargs = mock_normalizer.send_prompt_async.call_args[1]
            assert call_kwargs["labels"] == {"env": "staging"}

    @pytest.mark.asyncio
    async def test_add_message_raises_when_no_target_id(self, attack_service, mock_memory) -> None:
        """Test that add_message raises ValueError when attack has no target configured."""
        ar = make_attack_result(conversation_id="test-id", has_target=False)
        mock_memory.get_attack_results.return_value = [ar]

        request = AddMessageRequest(
            pieces=[MessagePieceRequest(original_value="Hello")],
        )

        with pytest.raises(ValueError, match="has no target configured"):
            await attack_service.add_message_async(conversation_id="test-id", request=request)

    @pytest.mark.asyncio
    async def test_add_message_with_send_calls_normalizer(self, attack_service, mock_memory) -> None:
        """Test that add_message with send=True sends message via normalizer."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []
        mock_memory.get_conversation.return_value = []

        with (
            patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_svc,
            patch("pyrit.backend.services.attack_service.PromptNormalizer") as mock_normalizer_cls,
        ):
            mock_target_svc = MagicMock()
            mock_target_svc.get_target_object.return_value = MagicMock()
            mock_get_target_svc.return_value = mock_target_svc

            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_cls.return_value = mock_normalizer

            request = AddMessageRequest(
                pieces=[MessagePieceRequest(original_value="Hello")],
                send=True,
            )

            result = await attack_service.add_message_async(conversation_id="test-id", request=request)

            mock_normalizer.send_prompt_async.assert_called_once()
            assert result.attack is not None

    @pytest.mark.asyncio
    async def test_add_message_with_send_raises_when_target_not_found(self, attack_service, mock_memory) -> None:
        """Test that add_message with send=True raises when target object not found."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []

        with patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_svc:
            mock_target_svc = MagicMock()
            mock_target_svc.get_target_object.return_value = None
            mock_get_target_svc.return_value = mock_target_svc

            request = AddMessageRequest(
                pieces=[MessagePieceRequest(original_value="Hello")],
                send=True,
            )

            with pytest.raises(ValueError, match="Target object .* not found"):
                await attack_service.add_message_async(conversation_id="test-id", request=request)

    @pytest.mark.asyncio
    async def test_add_message_with_converter_ids_gets_converters(self, attack_service, mock_memory) -> None:
        """Test that add_message with converter_ids gets converters from service."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []
        mock_memory.get_conversation.return_value = []

        with (
            patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_svc,
            patch("pyrit.backend.services.attack_service.get_converter_service") as mock_get_conv_svc,
            patch("pyrit.backend.services.attack_service.PromptNormalizer") as mock_normalizer_cls,
            patch("pyrit.backend.services.attack_service.PromptConverterConfiguration") as mock_config,
        ):
            mock_target_svc = MagicMock()
            mock_target_svc.get_target_object.return_value = MagicMock()
            mock_get_target_svc.return_value = mock_target_svc

            mock_conv_svc = MagicMock()
            mock_conv_svc.get_converter_objects_for_ids.return_value = [MagicMock()]
            mock_get_conv_svc.return_value = mock_conv_svc

            mock_config.from_converters.return_value = [MagicMock()]

            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_cls.return_value = mock_normalizer

            request = AddMessageRequest(
                pieces=[MessagePieceRequest(original_value="Hello")],
                send=True,
                converter_ids=["conv-1"],
            )

            await attack_service.add_message_async(conversation_id="test-id", request=request)

            mock_conv_svc.get_converter_objects_for_ids.assert_called_once_with(converter_ids=["conv-1"])

    @pytest.mark.asyncio
    async def test_add_message_raises_when_attack_not_found_after_update(self, attack_service, mock_memory) -> None:
        """Test that add_message raises ValueError when attack disappears after update."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []
        mock_memory.get_conversation.return_value = []

        request = AddMessageRequest(
            role="system",
            pieces=[MessagePieceRequest(original_value="Hello")],
            send=False,
        )

        with patch.object(attack_service, "get_attack_async", new=AsyncMock(return_value=None)):
            with pytest.raises(ValueError, match="not found after update"):
                await attack_service.add_message_async(conversation_id="test-id", request=request)

    @pytest.mark.asyncio
    async def test_add_message_raises_when_messages_not_found_after_update(self, attack_service, mock_memory) -> None:
        """Test that add_message raises ValueError when messages disappear after update."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []
        mock_memory.get_conversation.return_value = []

        request = AddMessageRequest(
            role="system",
            pieces=[MessagePieceRequest(original_value="Hello")],
            send=False,
        )

        with (
            patch.object(attack_service, "get_attack_async", new=AsyncMock(return_value=MagicMock())),
            patch.object(attack_service, "get_attack_messages_async", new=AsyncMock(return_value=None)),
        ):
            with pytest.raises(ValueError, match="messages not found after update"):
                await attack_service.add_message_async(conversation_id="test-id", request=request)

    @pytest.mark.asyncio
    async def test_get_converter_configs_skips_when_preconverted(self, attack_service, mock_memory) -> None:
        """Test that _get_converter_configs returns [] when pieces have converted_value set."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []
        mock_memory.get_conversation.return_value = []

        with (
            patch("pyrit.backend.services.attack_service.get_target_service") as mock_get_target_svc,
            patch("pyrit.backend.services.attack_service.get_converter_service") as mock_get_conv_svc,
            patch("pyrit.backend.services.attack_service.PromptNormalizer") as mock_normalizer_cls,
        ):
            mock_target_svc = MagicMock()
            mock_target_svc.get_target_object.return_value = MagicMock()
            mock_get_target_svc.return_value = mock_target_svc

            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_cls.return_value = mock_normalizer

            request = AddMessageRequest(
                pieces=[MessagePieceRequest(original_value="Hello", converted_value="SGVsbG8=")],
                send=True,
                converter_ids=["conv-1"],
            )

            await attack_service.add_message_async(conversation_id="test-id", request=request)

            # Converter service should NOT be called since pieces are preconverted
            mock_get_conv_svc.assert_not_called()
            # Normalizer should still be called with empty converter configs
            call_kwargs = mock_normalizer.send_prompt_async.call_args[1]
            assert call_kwargs["request_converter_configurations"] == []

    @pytest.mark.asyncio
    async def test_add_message_no_existing_pieces_labels_none(self, attack_service, mock_memory) -> None:
        """Test that add_message with no existing pieces passes labels=None to storage."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]
        mock_memory.get_message_pieces.return_value = []  # No existing pieces
        mock_memory.get_conversation.return_value = []

        request = AddMessageRequest(
            role="user",
            pieces=[MessagePieceRequest(original_value="Hello")],
            send=False,
        )

        result = await attack_service.add_message_async(conversation_id="test-id", request=request)

        stored_piece = mock_memory.add_message_pieces_to_memory.call_args[1]["message_pieces"][0]
        # No labels inherited from existing pieces (no existing pieces had labels)
        assert stored_piece.labels is None or stored_piece.labels == {}
        assert result.attack is not None


# ============================================================================
# Pagination Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestPagination:
    """Tests for pagination in list_attacks."""

    @pytest.mark.asyncio
    async def test_list_attacks_with_cursor_paginates(self, attack_service, mock_memory) -> None:
        """Test that list_attacks with cursor starts from the right position."""
        ar1 = make_attack_result(conversation_id="attack-1")
        ar2 = make_attack_result(conversation_id="attack-2")
        ar3 = make_attack_result(conversation_id="attack-3")
        mock_memory.get_attack_results.return_value = [ar1, ar2, ar3]
        mock_memory.get_message_pieces.return_value = []

        # Get first page
        result = await attack_service.list_attacks_async(limit=2)
        # Results are sorted by updated_at desc, so order may vary
        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_list_attacks_has_more_flag(self, attack_service, mock_memory) -> None:
        """Test that list_attacks sets has_more flag correctly."""
        ar1 = make_attack_result(conversation_id="attack-1")
        ar2 = make_attack_result(conversation_id="attack-2")
        ar3 = make_attack_result(conversation_id="attack-3")
        mock_memory.get_attack_results.return_value = [ar1, ar2, ar3]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(limit=2)

        assert result.pagination.has_more is True
        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_list_attacks_cursor_skips_to_correct_position(self, attack_service, mock_memory) -> None:
        """Test that list_attacks with cursor skips items before cursor."""
        ar1 = make_attack_result(
            conversation_id="attack-1",
            updated_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )
        ar2 = make_attack_result(
            conversation_id="attack-2",
            updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        ar3 = make_attack_result(
            conversation_id="attack-3",
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mock_memory.get_attack_results.return_value = [ar1, ar2, ar3]
        mock_memory.get_message_pieces.return_value = []

        # Cursor = attack-1 should skip attack-1 and return from attack-2 onward
        result = await attack_service.list_attacks_async(cursor="attack-1", limit=10)

        attack_ids = [item.conversation_id for item in result.items]
        assert "attack-1" not in attack_ids
        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_list_attacks_fetches_pieces_only_for_page(self, attack_service, mock_memory) -> None:
        """Test that pieces are fetched only for the paginated page, not all attacks."""
        attacks = [make_attack_result(conversation_id=f"attack-{i}") for i in range(5)]
        mock_memory.get_attack_results.return_value = attacks
        mock_memory.get_message_pieces.return_value = []

        await attack_service.list_attacks_async(limit=2)

        # get_message_pieces should be called only for the 2 items on the page, not all 5
        assert mock_memory.get_message_pieces.call_count == 2

    @pytest.mark.asyncio
    async def test_pagination_cursor_not_found_returns_from_start(self, attack_service, mock_memory) -> None:
        """Test that a stale/invalid cursor defaults to returning from the beginning."""
        ar1 = make_attack_result(
            conversation_id="attack-1",
            updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        ar2 = make_attack_result(
            conversation_id="attack-2",
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mock_memory.get_attack_results.return_value = [ar1, ar2]
        mock_memory.get_message_pieces.return_value = []

        result = await attack_service.list_attacks_async(cursor="nonexistent-cursor", limit=10)

        # Should return all items (from beginning) since cursor wasn't found
        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_pagination_cursor_at_last_item_returns_empty(self, attack_service, mock_memory) -> None:
        """Test that cursor pointing to the last item returns empty page with has_more=False."""
        ar1 = make_attack_result(
            conversation_id="attack-1",
            updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        ar2 = make_attack_result(
            conversation_id="attack-2",
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mock_memory.get_attack_results.return_value = [ar1, ar2]
        mock_memory.get_message_pieces.return_value = []

        # Cursor = last sorted item (attack-2 has the oldest updated_at, so it's last)
        result = await attack_service.list_attacks_async(cursor="attack-2", limit=10)

        assert len(result.items) == 0
        assert result.pagination.has_more is False


# ============================================================================
# Message Building Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestMessageBuilding:
    """Tests for message translation and building."""

    @pytest.mark.asyncio
    async def test_get_attack_with_messages_translates_correctly(self, attack_service, mock_memory) -> None:
        """Test that get_attack_messages translates PyRIT messages to backend format."""
        ar = make_attack_result(conversation_id="test-id")
        mock_memory.get_attack_results.return_value = [ar]

        # Create mock message with pieces
        mock_piece = MagicMock()
        mock_piece.id = "piece-1"
        mock_piece.converted_value_data_type = "text"
        mock_piece.original_value_data_type = "text"
        mock_piece.original_value = "Hello"
        mock_piece.converted_value = "Hello"
        mock_piece.response_error = None
        mock_piece.sequence = 0
        mock_piece.role = "user"
        mock_piece.get_role_for_storage.return_value = "user"
        mock_piece.timestamp = datetime.now(timezone.utc)
        mock_piece.scores = None

        mock_msg = MagicMock()
        mock_msg.message_pieces = [mock_piece]

        mock_memory.get_conversation.return_value = [mock_msg]

        result = await attack_service.get_attack_messages_async(conversation_id="test-id")

        assert result is not None
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert len(result.messages[0].pieces) == 1
        assert result.messages[0].pieces[0].original_value == "Hello"


# ============================================================================
# Singleton Tests
# ============================================================================


@pytest.mark.usefixtures("patch_central_database")
class TestAttackServiceSingleton:
    """Tests for get_attack_service singleton function."""

    def test_get_attack_service_returns_attack_service(self) -> None:
        """Test that get_attack_service returns an AttackService instance."""
        get_attack_service.cache_clear()

        with patch("pyrit.backend.services.attack_service.CentralMemory"):
            service = get_attack_service()
            assert isinstance(service, AttackService)

    def test_get_attack_service_returns_same_instance(self) -> None:
        """Test that get_attack_service returns the same instance."""
        get_attack_service.cache_clear()

        with patch("pyrit.backend.services.attack_service.CentralMemory"):
            service1 = get_attack_service()
            service2 = get_attack_service()
            assert service1 is service2
