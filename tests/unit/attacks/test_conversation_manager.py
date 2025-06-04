# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.components.conversation_manager import (
    ConversationManager,
    ConversationState,
)
from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget


@pytest.fixture
def mock_memory():
    """Create a mock memory instance for testing"""
    memory = MagicMock(spec=MemoryInterface)
    memory.get_conversation.return_value = []
    memory.get_scores_by_prompt_ids.return_value = []
    memory.add_request_response_to_memory = MagicMock()
    return memory


@pytest.fixture
def mock_prompt_normalizer():
    """Create a mock prompt normalizer for testing"""
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.convert_values = AsyncMock()
    return normalizer


@pytest.fixture
def mock_chat_target():
    """Create a mock chat target for testing"""
    target = MagicMock(spec=PromptChatTarget)
    target.set_system_prompt = MagicMock()
    target.get_identifier.return_value = {"id": "mock_chat_target_id"}
    return target


@pytest.fixture
def mock_prompt_target():
    """Create a mock prompt target (non-chat) for testing"""
    target = MagicMock(spec=PromptTarget)
    target.get_identifier.return_value = {"id": "mock_target_id"}
    return target


@pytest.fixture
def attack_identifier():
    """Create a sample attack identifier"""
    return {"attack_id": str(uuid.uuid4()), "type": "test_attack"}


@pytest.fixture
def sample_user_piece():
    """Create a sample user prompt request piece"""
    return PromptRequestPiece(
        role="user",
        original_value="Hello, how are you?",
        original_value_data_type="text",
        conversation_id=str(uuid.uuid4()),
    )


@pytest.fixture
def sample_assistant_piece():
    """Create a sample assistant prompt request piece"""
    return PromptRequestPiece(
        role="assistant",
        original_value="I'm doing well, thank you!",
        original_value_data_type="text",
        conversation_id=str(uuid.uuid4()),
    )


@pytest.fixture
def sample_system_piece():
    """Create a sample system prompt request piece"""
    return PromptRequestPiece(
        role="system",
        original_value="You are a helpful assistant",
        original_value_data_type="text",
        conversation_id=str(uuid.uuid4()),
    )


@pytest.fixture
def sample_conversation(sample_user_piece, sample_assistant_piece):
    """Create a sample conversation with user and assistant messages"""
    return [
        PromptRequestResponse(request_pieces=[sample_user_piece]),
        PromptRequestResponse(request_pieces=[sample_assistant_piece]),
    ]


@pytest.fixture
def sample_score():
    """Create a sample score for testing"""
    return Score(
        score_type="true_false",
        score_value="true",
        score_category="test",
        score_value_description="Test score",
        score_rationale="Test rationale",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestConversationManagerInitialization:
    """Tests for ConversationManager initialization"""

    def test_init_with_required_parameters(self, attack_identifier):
        manager = ConversationManager(attack_identifier=attack_identifier)

        assert manager._attack_identifier == attack_identifier
        assert isinstance(manager._prompt_normalizer, PromptNormalizer)
        assert manager._memory is not None

    def test_init_with_custom_prompt_normalizer(self, attack_identifier, mock_prompt_normalizer):
        manager = ConversationManager(attack_identifier=attack_identifier, prompt_normalizer=mock_prompt_normalizer)

        assert manager._prompt_normalizer == mock_prompt_normalizer


@pytest.mark.usefixtures("patch_central_database")
class TestConversationRetrieval:
    """Tests for conversation retrieval methods"""

    def test_get_conversation_returns_empty_list_when_no_messages(self, attack_identifier, mock_memory):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            # Ensure the manager is using our mock memory
            manager._memory = mock_memory

            mock_memory.get_conversation.return_value = []
            conversation_id = str(uuid.uuid4())

            result = manager.get_conversation(conversation_id)

            assert result == []
            mock_memory.get_conversation.assert_called_once_with(conversation_id=conversation_id)

    def test_get_conversation_returns_messages_in_order(self, attack_identifier, mock_memory, sample_conversation):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_conversation.return_value = sample_conversation
            conversation_id = str(uuid.uuid4())

            result = manager.get_conversation(conversation_id)

            assert result == sample_conversation
            assert len(result) == 2

    def test_get_last_message_returns_none_for_empty_conversation(self, attack_identifier, mock_memory):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_conversation.return_value = []
            conversation_id = str(uuid.uuid4())

            result = manager.get_last_message(conversation_id=conversation_id)

            assert result is None

    def test_get_last_message_returns_last_piece(self, attack_identifier, mock_memory, sample_conversation):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_conversation.return_value = sample_conversation
            conversation_id = str(uuid.uuid4())

            result = manager.get_last_message(conversation_id=conversation_id)

            assert result == sample_conversation[-1].get_piece()
            assert result is not None
            assert result.role == "assistant"

    def test_get_last_message_with_role_filter(self, attack_identifier, mock_memory, sample_conversation):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_conversation.return_value = sample_conversation
            conversation_id = str(uuid.uuid4())

            # Get last user message
            result = manager.get_last_message(conversation_id=conversation_id, role="user")

            assert result == sample_conversation[0].get_piece()
            assert result is not None
            assert result.role == "user"

    def test_get_last_message_with_role_filter_returns_none_when_no_match(
        self, attack_identifier, mock_memory, sample_conversation
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_conversation.return_value = sample_conversation
            conversation_id = str(uuid.uuid4())

            # Try to get system message when none exists
            result = manager.get_last_message(conversation_id=conversation_id, role="system")

            assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestSystemPromptHandling:
    """Tests for system prompt functionality"""

    def test_add_system_prompt_with_chat_target(self, attack_identifier, mock_memory, mock_chat_target):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())
            system_prompt = "You are a helpful assistant"
            labels = {"type": "system"}

            manager.add_system_prompt(
                target=mock_chat_target, conversation_id=conversation_id, system_prompt=system_prompt, labels=labels
            )

            mock_chat_target.set_system_prompt.assert_called_once_with(
                system_prompt=system_prompt,
                conversation_id=conversation_id,
                orchestrator_identifier=attack_identifier,
                labels=labels,
            )

    def test_add_system_prompt_raises_error_for_non_chat_target(
        self, attack_identifier, mock_memory, mock_prompt_target
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            with pytest.raises(ValueError, match="Objective Target must be a PromptChatTarget"):
                manager.add_system_prompt(
                    target=mock_prompt_target, conversation_id=conversation_id, system_prompt="System prompt"
                )

    def test_add_system_prompt_without_labels(self, attack_identifier, mock_memory, mock_chat_target):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())
            system_prompt = "You are a helpful assistant"

            manager.add_system_prompt(
                target=mock_chat_target, conversation_id=conversation_id, system_prompt=system_prompt
            )

            mock_chat_target.set_system_prompt.assert_called_once()
            call_args = mock_chat_target.set_system_prompt.call_args
            assert call_args.kwargs["labels"] is None


@pytest.mark.usefixtures("patch_central_database")
class TestConversationStateUpdate:
    """Tests for conversation state update functionality"""

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_empty_conversation_id_raises_error(
        self, attack_identifier, mock_memory
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            with pytest.raises(ValueError, match="conversation_id cannot be empty"):
                await manager.update_conversation_state_async(conversation_id="", prepended_conversation=[])

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_empty_history_returns_default_state(
        self, attack_identifier, mock_memory
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=[]
            )

            assert isinstance(state, ConversationState)
            assert state.turn_count == 1
            assert state.last_user_message == ""
            assert state.last_assistant_message_scores == []

    @pytest.mark.asyncio
    async def test_update_conversation_state_single_turn_mode(
        self, attack_identifier, mock_memory, sample_conversation
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Single-turn mode (no max_turns)
            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=sample_conversation
            )

            # Verify all messages were added to memory
            assert mock_memory.add_request_response_to_memory.call_count == 2

            # State should remain default in single-turn mode
            assert state.turn_count == 1
            assert state.last_user_message == ""

    @pytest.mark.asyncio
    async def test_update_conversation_state_multi_turn_mode_excludes_last_user_message(
        self, attack_identifier, mock_memory, sample_user_piece
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Create conversation ending with user message
            conversation = [PromptRequestResponse(request_pieces=[sample_user_piece])]

            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
            )

            # Last user message should be excluded from memory in multi-turn mode
            mock_memory.add_request_response_to_memory.assert_not_called()

            # But should be captured in state
            assert state.last_user_message == sample_user_piece.converted_value

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_converters(
        self, attack_identifier, mock_memory, mock_prompt_normalizer, sample_conversation
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier, prompt_normalizer=mock_prompt_normalizer)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())
            converter_config = [PromptConverterConfiguration(converters=[])]

            await manager.update_conversation_state_async(
                conversation_id=conversation_id,
                prepended_conversation=sample_conversation,
                converter_configurations=converter_config,
            )

            # Verify converters were applied to each message
            assert mock_prompt_normalizer.convert_values.call_count == 2
            for call in mock_prompt_normalizer.convert_values.call_args_list:
                assert call.kwargs["converter_configurations"] == converter_config

    @pytest.mark.asyncio
    async def test_update_conversation_state_processes_system_prompts(
        self, attack_identifier, mock_memory, mock_chat_target, sample_system_piece
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Create conversation with system message
            conversation = [PromptRequestResponse(request_pieces=[sample_system_piece])]

            await manager.update_conversation_state_async(
                conversation_id=conversation_id,
                prepended_conversation=conversation,
                target=mock_chat_target,
                max_turns=5,  # Multi-turn mode to trigger system prompt handling
            )

            # System prompt should be set on target
            mock_chat_target.set_system_prompt.assert_called_once()

            # System messages should not be added to memory in multi-turn mode
            mock_memory.add_request_response_to_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_conversation_state_counts_turns_correctly(
        self, attack_identifier, mock_memory, sample_user_piece, sample_assistant_piece
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Create multi-turn conversation
            conversation = [
                PromptRequestResponse(request_pieces=[sample_user_piece]),
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
                PromptRequestResponse(request_pieces=[sample_user_piece]),
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
            ]

            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
            )

            # Should count 2 assistant messages = 2 turns
            assert state.turn_count == 3  # Starts at 1, plus 2 assistant messages

    @pytest.mark.asyncio
    async def test_update_conversation_state_exceeds_max_turns_raises_error(
        self, attack_identifier, mock_memory, sample_user_piece, sample_assistant_piece
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Create conversation that exceeds max turns
            conversation = [
                PromptRequestResponse(request_pieces=[sample_user_piece]),
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
                PromptRequestResponse(request_pieces=[sample_user_piece]),
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
            ]

            with pytest.raises(ValueError, match="exceeds the maximum number of turns"):
                await manager.update_conversation_state_async(
                    conversation_id=conversation_id,
                    prepended_conversation=conversation,
                    max_turns=1,  # Only allow 1 turn
                )

    @pytest.mark.asyncio
    async def test_update_conversation_state_extracts_assistant_scores(
        self, attack_identifier, mock_memory, sample_user_piece, sample_assistant_piece, sample_score
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_scores_by_prompt_ids.return_value = [sample_score]
            conversation_id = str(uuid.uuid4())

            # Create conversation ending with assistant message
            conversation = [
                PromptRequestResponse(request_pieces=[sample_user_piece]),
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
            ]

            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
            )

            # Should extract scores for last assistant message
            assert state.last_assistant_message_scores == [sample_score]
            assert state.last_user_message == sample_user_piece.converted_value

    @pytest.mark.asyncio
    async def test_update_conversation_state_no_scores_for_assistant_message(
        self, attack_identifier, mock_memory, sample_assistant_piece
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_scores_by_prompt_ids.return_value = []
            conversation_id = str(uuid.uuid4())

            # Create conversation with only assistant message
            conversation = [
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
            ]

            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
            )

            # Should not set last_user_message when no scores found
            assert state.last_assistant_message_scores == []
            assert state.last_user_message == ""

    @pytest.mark.asyncio
    async def test_update_conversation_state_assistant_without_preceding_user_raises_error(
        self, attack_identifier, mock_memory, sample_assistant_piece, sample_score
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            mock_memory.get_scores_by_prompt_ids.return_value = [sample_score]
            conversation_id = str(uuid.uuid4())

            # Create conversation with assistant messages only
            conversation = [
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
                PromptRequestResponse(request_pieces=[sample_assistant_piece]),
            ]

            with pytest.raises(ValueError, match="There must be a user message preceding"):
                await manager.update_conversation_state_async(
                    conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
                )


@pytest.mark.usefixtures("patch_central_database")
class TestPrivateMethods:
    """Tests for private helper methods"""

    def test_should_exclude_piece_from_memory_single_turn_mode(self, sample_system_piece):
        # In single-turn mode (max_turns=None), nothing should be excluded
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_system_piece, max_turns=None)

    def test_should_exclude_piece_from_memory_multi_turn_system_piece(self, sample_system_piece):
        # In multi-turn mode, system pieces should be excluded
        assert ConversationManager._should_exclude_piece_from_memory(piece=sample_system_piece, max_turns=5)

    def test_should_exclude_piece_from_memory_multi_turn_non_system_piece(
        self, sample_user_piece, sample_assistant_piece
    ):
        # In multi-turn mode, non-system pieces should not be excluded
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_user_piece, max_turns=5)
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_assistant_piece, max_turns=5)


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_empty_request_pieces(self, attack_identifier, mock_memory):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Create request with empty pieces list
            conversation = [PromptRequestResponse(request_pieces=[])]

            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation
            )

            # Should handle gracefully
            assert state.turn_count == 1
            mock_memory.add_request_response_to_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_none_request(self, attack_identifier, mock_memory):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Create conversation with None values
            conversation = [None]  # This would be caught by type checking in real code

            # Should handle gracefully
            state = await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation  # type: ignore
            )

            assert state.turn_count == 1

    @pytest.mark.asyncio
    async def test_update_conversation_state_preserves_piece_metadata(
        self, attack_identifier, mock_memory, sample_user_piece
    ):
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory):
            manager = ConversationManager(attack_identifier=attack_identifier)
            manager._memory = mock_memory

            conversation_id = str(uuid.uuid4())

            # Add metadata to piece
            sample_user_piece.labels = {"test": "label"}
            sample_user_piece.prompt_metadata = {"key": "value"}

            conversation = [PromptRequestResponse(request_pieces=[sample_user_piece])]

            await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation
            )

            # Verify piece was processed with metadata intact
            call_args = mock_memory.add_request_response_to_memory.call_args
            processed_piece = call_args.kwargs["request"].request_pieces[0]
            assert processed_piece.labels == {"test": "label"}
            assert processed_piece.prompt_metadata == {"key": "value"}

    def test_conversation_state_dataclass_defaults(self):
        # Test ConversationState initialization with defaults
        state = ConversationState()

        assert state.turn_count == 1
        assert state.last_user_message == ""
        assert state.last_assistant_message_scores == []

    def test_conversation_state_dataclass_with_values(self, sample_score):
        # Test ConversationState initialization with custom values
        state = ConversationState(
            turn_count=5, last_user_message="Test message", last_assistant_message_scores=[sample_score]
        )

        assert state.turn_count == 5
        assert state.last_user_message == "Test message"
        assert state.last_assistant_message_scores == [sample_score]
