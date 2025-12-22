# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack import ConversationManager
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import PromptChatTarget


@pytest.fixture
def mock_adversarial_chat() -> MagicMock:
    """Create a mock adversarial chat target for testing."""
    chat = MagicMock(spec=PromptChatTarget)
    chat.set_system_prompt = MagicMock()
    chat.get_identifier.return_value = {"__type__": "MockAdversarialChat", "__module__": "test_module"}
    return chat


@pytest.fixture
def attack_identifier() -> dict[str, str]:
    """Create a sample attack identifier."""
    return {
        "__type__": "TestAttack",
        "__module__": "pyrit.executor.attack.test_attack",
        "id": str(uuid.uuid4()),
    }


@pytest.fixture
def sample_user_piece() -> MessagePiece:
    """Create a sample user message piece."""
    return MessagePiece(
        role="user",
        original_value="Hello, how are you?",
        original_value_data_type="text",
        converted_value="Hello, how are you?",
        converted_value_data_type="text",
        conversation_id=str(uuid.uuid4()),
    )


@pytest.fixture
def sample_assistant_piece() -> MessagePiece:
    """Create a sample assistant message piece."""
    return MessagePiece(
        role="assistant",
        original_value="I'm doing well, thank you!",
        original_value_data_type="text",
        converted_value="I'm doing well, thank you!",
        converted_value_data_type="text",
        conversation_id=str(uuid.uuid4()),
    )


@pytest.fixture
def sample_system_piece() -> MessagePiece:
    """Create a sample system message piece."""
    return MessagePiece(
        role="system",
        original_value="You are a helpful assistant",
        original_value_data_type="text",
        converted_value="You are a helpful assistant",
        converted_value_data_type="text",
        conversation_id=str(uuid.uuid4()),
    )


@pytest.fixture
def sample_prepended_conversation(
    sample_user_piece: MessagePiece, sample_assistant_piece: MessagePiece
) -> list[Message]:
    """Create a sample prepended conversation with user and assistant messages."""
    return [
        Message(message_pieces=[sample_user_piece]),
        Message(message_pieces=[sample_assistant_piece]),
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestPrependToAdversarialChat:
    """Tests for prepend_to_adversarial_chat_async functionality."""

    @pytest.mark.asyncio
    async def test_prepend_empty_conversation_does_nothing(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
    ):
        """Test that empty prepended conversation doesn't add any messages."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=[],
        )

        # Verify no messages were added
        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 0

    @pytest.mark.asyncio
    async def test_prepend_swaps_user_to_assistant_role(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_user_piece: MessagePiece,
    ):
        """Test that user messages become assistant messages for adversarial chat."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        prepended = [Message(message_pieces=[sample_user_piece])]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=prepended,
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 1
        # User role should be swapped to assistant
        assert stored[0].message_pieces[0].role == "assistant"
        assert stored[0].message_pieces[0].original_value == sample_user_piece.original_value

    @pytest.mark.asyncio
    async def test_prepend_swaps_assistant_to_user_role(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_assistant_piece: MessagePiece,
    ):
        """Test that assistant messages become user messages for adversarial chat."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        prepended = [Message(message_pieces=[sample_assistant_piece])]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=prepended,
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 1
        # Assistant role should be swapped to user
        assert stored[0].message_pieces[0].role == "user"
        assert stored[0].message_pieces[0].original_value == sample_assistant_piece.original_value

    @pytest.mark.asyncio
    async def test_prepend_skips_system_messages(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_system_piece: MessagePiece,
        sample_user_piece: MessagePiece,
    ):
        """Test that system messages are skipped when prepending to adversarial chat."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        prepended = [
            Message(message_pieces=[sample_system_piece]),
            Message(message_pieces=[sample_user_piece]),
        ]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=prepended,
        )

        stored = manager.get_conversation(conversation_id)
        # Only the user message should be stored (as assistant), system skipped
        assert len(stored) == 1
        assert stored[0].message_pieces[0].role == "assistant"

    @pytest.mark.asyncio
    async def test_prepend_multi_turn_conversation_swaps_all_roles(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_prepended_conversation: list[Message],
    ):
        """Test that a multi-turn conversation has all roles properly swapped."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=sample_prepended_conversation,
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 2
        # First message (was user) should now be assistant
        assert stored[0].message_pieces[0].role == "assistant"
        # Second message (was assistant) should now be user
        assert stored[1].message_pieces[0].role == "user"

    @pytest.mark.asyncio
    async def test_prepend_preserves_message_content(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_prepended_conversation: list[Message],
    ):
        """Test that message content is preserved after role swap."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        original_values = [
            msg.message_pieces[0].original_value for msg in sample_prepended_conversation
        ]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=sample_prepended_conversation,
        )

        stored = manager.get_conversation(conversation_id)
        stored_values = [msg.message_pieces[0].original_value for msg in stored]
        assert stored_values == original_values

    @pytest.mark.asyncio
    async def test_prepend_uses_correct_conversation_id(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_user_piece: MessagePiece,
    ):
        """Test that prepended messages use the adversarial chat conversation ID."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        prepended = [Message(message_pieces=[sample_user_piece])]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=prepended,
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 1
        assert stored[0].message_pieces[0].conversation_id == conversation_id

    @pytest.mark.asyncio
    async def test_prepend_sets_target_identifier(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_user_piece: MessagePiece,
    ):
        """Test that prepended messages have the adversarial chat's target identifier."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        prepended = [Message(message_pieces=[sample_user_piece])]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=prepended,
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 1
        assert stored[0].message_pieces[0].prompt_target_identifier == mock_adversarial_chat.get_identifier()

    @pytest.mark.asyncio
    async def test_prepend_applies_labels(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
        sample_user_piece: MessagePiece,
    ):
        """Test that labels are applied to prepended messages."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())
        labels = {"source": "test", "type": "prepended"}

        prepended = [Message(message_pieces=[sample_user_piece])]

        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=prepended,
            labels=labels,
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 1
        assert stored[0].message_pieces[0].labels == labels

    @pytest.mark.asyncio
    async def test_prepend_none_conversation_does_nothing(
        self,
        attack_identifier: dict[str, str],
        mock_adversarial_chat: MagicMock,
    ):
        """Test that None prepended conversation is handled gracefully."""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Should not raise, just return early
        await manager.prepend_to_adversarial_chat_async(
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_conversation_id=conversation_id,
            prepended_conversation=None,  # type: ignore
        )

        stored = manager.get_conversation(conversation_id)
        assert len(stored) == 0
