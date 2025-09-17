# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack import (
    ConversationManager,
    ConversationState,
)
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget


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
    return {
        "__type__": "TestAttack",
        "__module__": "pyrit.executor.attack.test_attack",
        "id": str(uuid.uuid4()),
    }


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
def sample_conversation(sample_user_piece: PromptRequestPiece, sample_assistant_piece: PromptRequestPiece):
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

    def test_init_with_required_parameters(self, attack_identifier: dict[str, str]):
        manager = ConversationManager(attack_identifier=attack_identifier)

        assert manager._attack_identifier == attack_identifier
        assert isinstance(manager._prompt_normalizer, PromptNormalizer)
        assert manager._memory is not None

    def test_init_with_custom_prompt_normalizer(
        self, attack_identifier: dict[str, str], mock_prompt_normalizer: MagicMock
    ):
        manager = ConversationManager(attack_identifier=attack_identifier, prompt_normalizer=mock_prompt_normalizer)

        assert manager._prompt_normalizer == mock_prompt_normalizer


@pytest.mark.usefixtures("patch_central_database")
class TestConversationRetrieval:
    """Tests for conversation retrieval methods"""

    def test_get_conversation_returns_empty_list_when_no_messages(self, attack_identifier: dict[str, str]):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        result = manager.get_conversation(conversation_id)

        assert result == []

    def test_get_conversation_returns_messages_in_order(
        self, attack_identifier: dict[str, str], sample_conversation: list[PromptRequestResponse]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Add messages to the database
        for response in sample_conversation:
            for piece in response.request_pieces:
                piece.conversation_id = conversation_id
            manager._memory.add_request_response_to_memory(request=response)

        result = manager.get_conversation(conversation_id)

        assert len(result) == 2
        assert result[0].request_pieces[0].role == "user"
        assert result[1].request_pieces[0].role == "assistant"

    def test_get_last_message_returns_none_for_empty_conversation(self, attack_identifier: dict[str, str]):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        result = manager.get_last_message(conversation_id=conversation_id)

        assert result is None

    def test_get_last_message_returns_last_piece(
        self, attack_identifier: dict[str, str], sample_conversation: list[PromptRequestResponse]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Add messages to the database
        for response in sample_conversation:
            for piece in response.request_pieces:
                piece.conversation_id = conversation_id
            manager._memory.add_request_response_to_memory(request=response)

        result = manager.get_last_message(conversation_id=conversation_id)

        assert result is not None
        assert result.role == "assistant"

    def test_get_last_message_with_role_filter(
        self, attack_identifier: dict[str, str], sample_conversation: list[PromptRequestResponse]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Add messages to the database
        for response in sample_conversation:
            for piece in response.request_pieces:
                piece.conversation_id = conversation_id
            manager._memory.add_request_response_to_memory(request=response)

        # Get last user message
        result = manager.get_last_message(conversation_id=conversation_id, role="user")

        assert result is not None
        assert result.role == "user"

    def test_get_last_message_with_role_filter_returns_none_when_no_match(
        self, attack_identifier: dict[str, str], sample_conversation: list[PromptRequestResponse]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Add messages to the database
        for response in sample_conversation:
            for piece in response.request_pieces:
                piece.conversation_id = conversation_id
            manager._memory.add_request_response_to_memory(request=response)

        # Try to get system message when none exists
        result = manager.get_last_message(conversation_id=conversation_id, role="system")

        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestSystemPromptHandling:
    """Tests for system prompt functionality"""

    def test_set_system_prompt_with_chat_target(self, attack_identifier: dict[str, str], mock_chat_target: MagicMock):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())
        system_prompt = "You are a helpful assistant"
        labels = {"type": "system"}

        manager.set_system_prompt(
            target=mock_chat_target, conversation_id=conversation_id, system_prompt=system_prompt, labels=labels
        )

        mock_chat_target.set_system_prompt.assert_called_once_with(
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            attack_identifier=attack_identifier,
            labels=labels,
        )

    def test_set_system_prompt_raises_error_for_non_chat_target(
        self, attack_identifier: dict[str, str], mock_prompt_target: MagicMock
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        with pytest.raises(AttributeError, match="Mock object has no attribute 'set_system_prompt'"):
            manager.set_system_prompt(
                target=mock_prompt_target, conversation_id=conversation_id, system_prompt="System prompt"
            )

    def test_set_system_prompt_without_labels(self, attack_identifier: dict[str, str], mock_chat_target: MagicMock):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())
        system_prompt = "You are a helpful assistant"

        manager.set_system_prompt(target=mock_chat_target, conversation_id=conversation_id, system_prompt=system_prompt)

        mock_chat_target.set_system_prompt.assert_called_once()
        call_args = mock_chat_target.set_system_prompt.call_args
        assert call_args.kwargs["labels"] is None


@pytest.mark.usefixtures("patch_central_database")
class TestConversationStateUpdate:
    """Tests for conversation state update functionality"""

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_empty_conversation_id_raises_error(
        self, attack_identifier: dict[str, str]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)

        with pytest.raises(ValueError, match="conversation_id cannot be empty"):
            await manager.update_conversation_state_async(conversation_id="", prepended_conversation=[])

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_empty_history_returns_default_state(
        self, attack_identifier: dict[str, str]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        state = await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=[]
        )

        assert isinstance(state, ConversationState)
        assert state.turn_count == 0
        assert state.last_user_message == ""
        assert state.last_assistant_message_scores == []

    @pytest.mark.asyncio
    async def test_update_conversation_state_single_turn_mode(
        self, attack_identifier: dict[str, str], sample_conversation: list[PromptRequestResponse]
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Single-turn mode (no max_turns)
        state = await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=sample_conversation
        )

        # Verify all messages were added to memory
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 2

        # State should remain default in single-turn mode
        assert state.turn_count == 0
        assert state.last_user_message == ""

    @pytest.mark.asyncio
    async def test_update_conversation_state_multi_turn_mode_excludes_last_user_message(
        self, attack_identifier: dict[str, str], sample_user_piece: PromptRequestPiece
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation ending with user message
        conversation = [PromptRequestResponse(request_pieces=[sample_user_piece])]

        state = await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
        )

        # Last user message should be excluded from memory in multi-turn mode
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 0

        # But should be captured in state
        assert state.last_user_message == sample_user_piece.converted_value

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_role_specific_converters(
        self,
        attack_identifier: dict[str, str],
        mock_prompt_normalizer: MagicMock,
        sample_conversation: list[PromptRequestResponse],
    ):
        """Test that role-specific converters apply correctly"""
        manager = ConversationManager(attack_identifier=attack_identifier, prompt_normalizer=mock_prompt_normalizer)
        conversation_id = str(uuid.uuid4())

        request_converter_config = [PromptConverterConfiguration(converters=[])]
        response_converter_config = [PromptConverterConfiguration(converters=[])]

        await manager.update_conversation_state_async(
            conversation_id=conversation_id,
            prepended_conversation=sample_conversation,
            request_converters=request_converter_config,
            response_converters=response_converter_config,
        )

        # Verify converters were applied to both user and assistant messages
        assert mock_prompt_normalizer.convert_values.call_count == 2

        # First call should be for user message with request converters
        user_call = mock_prompt_normalizer.convert_values.call_args_list[0]
        assert user_call.kwargs["converter_configurations"] == request_converter_config

        # Second call should be for assistant message with response converters
        assistant_call = mock_prompt_normalizer.convert_values.call_args_list[1]
        assert assistant_call.kwargs["converter_configurations"] == response_converter_config
        assert mock_prompt_normalizer.convert_values.call_count == 2

        # Check that the right converters were applied to the right roles
        calls = mock_prompt_normalizer.convert_values.call_args_list
        # First call should be for user message with request converters
        assert calls[0].kwargs["converter_configurations"] == request_converter_config
        # Second call should be for assistant message with response converters
        assert calls[1].kwargs["converter_configurations"] == response_converter_config

    @pytest.mark.asyncio
    async def test_update_conversation_state_system_messages_no_converters(
        self,
        attack_identifier: dict[str, str],
        mock_prompt_normalizer: MagicMock,
        mock_chat_target: MagicMock,
        sample_system_piece: PromptRequestPiece,
    ):
        """Test that system messages do not get converters applied regardless of config"""
        manager = ConversationManager(attack_identifier=attack_identifier, prompt_normalizer=mock_prompt_normalizer)
        conversation_id = str(uuid.uuid4())

        # Create conversation with just a system message
        conversation = [PromptRequestResponse(request_pieces=[sample_system_piece])]

        request_converter_config = [PromptConverterConfiguration(converters=[])]
        response_converter_config = [PromptConverterConfiguration(converters=[])]

        await manager.update_conversation_state_async(
            conversation_id=conversation_id,
            prepended_conversation=conversation,
            target=mock_chat_target,
            request_converters=request_converter_config,
            response_converters=response_converter_config,
            max_turns=5,  # Multi-turn mode to trigger system prompt handling
        )

        # No converters should be applied to system messages
        assert mock_prompt_normalizer.convert_values.call_count == 0

    @pytest.mark.asyncio
    async def test_update_conversation_state_processes_system_prompts_multi_turn(
        self, attack_identifier: dict[str, str], mock_chat_target: MagicMock, sample_system_piece: PromptRequestPiece
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
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
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 0

    @pytest.mark.asyncio
    async def test_update_conversation_state_processes_system_prompts_single_turn(
        self, attack_identifier: dict[str, str], mock_chat_target: MagicMock, sample_system_piece: PromptRequestPiece
    ):
        """Test that system messages in single-turn mode are NOT added to memory"""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation with system message
        conversation = [PromptRequestResponse(request_pieces=[sample_system_piece])]

        await manager.update_conversation_state_async(
            conversation_id=conversation_id,
            prepended_conversation=conversation,
            target=mock_chat_target,
            # No max_turns = single-turn mode
        )

        # System prompt should be set on target
        mock_chat_target.set_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_conversation_state_single_turn_behavior_matches_legacy(
        self,
        attack_identifier: dict[str, str],
        mock_chat_target: MagicMock,
        sample_user_piece: PromptRequestPiece,
        sample_assistant_piece: PromptRequestPiece,
        sample_system_piece: PromptRequestPiece,
    ):
        """Test that single-turn behavior correctly excludes system messages from memory"""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation with all types of messages
        conversation = [
            PromptRequestResponse(request_pieces=[sample_system_piece]),
            PromptRequestResponse(request_pieces=[sample_user_piece]),
            PromptRequestResponse(request_pieces=[sample_assistant_piece]),
        ]

        # Store original IDs to verify they get updated
        # Since we are mocking the target, the system piece won't be stored, so we only check user and assistant
        original_user_id = sample_user_piece.id
        original_assistant_id = sample_assistant_piece.id

        await manager.update_conversation_state_async(
            conversation_id=conversation_id,
            prepended_conversation=conversation,
            target=mock_chat_target,
            # No max_turns = single-turn mode
        )

        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 2

        # Verify that user and assistant pieces have the correct conversation_id and attack_identifier
        for stored_response in stored_conversation:
            for piece in stored_response.request_pieces:
                assert piece.conversation_id == conversation_id
                assert piece.attack_identifier == attack_identifier
                # Verify that IDs were regenerated
                assert piece.id != original_user_id
                assert piece.id != original_assistant_id
                # System piece should not be in memory, since we mocked the target

        # Verify roles are preserved and in order (excluding system)
        assert stored_conversation[0].get_piece().role == "user"
        assert stored_conversation[1].get_piece().role == "assistant"

        # System prompt should still be set on target even in single-turn mode
        mock_chat_target.set_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_conversation_state_system_prompt_without_target_raises_error(
        self, attack_identifier: dict[str, str], sample_system_piece: PromptRequestPiece
    ):
        """Test that providing system prompts without a target raises an error"""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation with system message
        conversation = [PromptRequestResponse(request_pieces=[sample_system_piece])]

        with pytest.raises(ValueError, match="Target must be provided to handle system prompts"):
            await manager.update_conversation_state_async(
                conversation_id=conversation_id,
                prepended_conversation=conversation,
                # No target provided
            )

    @pytest.mark.asyncio
    async def test_update_conversation_state_system_prompt_with_non_chat_target_raises_error(
        self, attack_identifier: dict[str, str], mock_prompt_target: MagicMock, sample_system_piece: PromptRequestPiece
    ):
        """Test that providing system prompts with non-chat target raises an error"""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation with system message
        conversation = [PromptRequestResponse(request_pieces=[sample_system_piece])]

        with pytest.raises(ValueError, match="Target must be a PromptChatTarget to set system prompts"):
            await manager.update_conversation_state_async(
                conversation_id=conversation_id,
                prepended_conversation=conversation,
                target=mock_prompt_target,  # Non-chat target
            )

    @pytest.mark.asyncio
    async def test_update_conversation_state_mixed_conversation_multi_turn(
        self,
        attack_identifier: dict[str, str],
        mock_chat_target: MagicMock,
        sample_user_piece: PromptRequestPiece,
        sample_assistant_piece: PromptRequestPiece,
        sample_system_piece: PromptRequestPiece,
    ):
        """Test that in multi-turn mode, system prompts are excluded but other messages are added"""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation with mixed message types
        conversation = [
            PromptRequestResponse(request_pieces=[sample_system_piece]),
            PromptRequestResponse(request_pieces=[sample_user_piece]),
            PromptRequestResponse(request_pieces=[sample_assistant_piece]),
        ]

        await manager.update_conversation_state_async(
            conversation_id=conversation_id,
            prepended_conversation=conversation,
            target=mock_chat_target,
            max_turns=5,  # Multi-turn mode
        )

        # System prompt should be set on target
        mock_chat_target.set_system_prompt.assert_called_once()

        # Only user and assistant messages should be in memory
        # Since the target is mocked, the system piece won't be stored
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 2
        assert stored_conversation[0].get_piece().role == "user"
        assert stored_conversation[1].get_piece().role == "assistant"

    @pytest.mark.asyncio
    async def test_update_conversation_state_preserves_original_values_like_legacy(
        self,
        attack_identifier: dict[str, str],
        sample_user_piece: PromptRequestPiece,
    ):
        """Test that original values and other piece properties are preserved like the legacy function"""
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Set up piece with various properties that should be preserved
        sample_user_piece.original_value = "Original user message"
        sample_user_piece.converted_value = "Converted user message"
        sample_user_piece.labels = {"test": "label", "category": "user"}
        sample_user_piece.prompt_metadata = {"timestamp": "2023-01-01", "version": 1}

        conversation = [PromptRequestResponse(request_pieces=[sample_user_piece])]

        await manager.update_conversation_state_async(
            conversation_id=conversation_id,
            prepended_conversation=conversation,
            # Single-turn mode
        )

        # Verify piece was stored with all properties preserved
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 1

        stored_piece = stored_conversation[0].get_piece()
        assert stored_piece.original_value == "Original user message"
        assert stored_piece.converted_value == "Converted user message"
        assert stored_piece.labels == {"test": "label", "category": "user"}
        assert stored_piece.prompt_metadata == {"timestamp": "2023-01-01", "version": 1}
        assert stored_piece.conversation_id == conversation_id
        assert stored_piece.attack_identifier == attack_identifier

    @pytest.mark.asyncio
    async def test_update_conversation_state_counts_turns_correctly(
        self,
        attack_identifier: dict[str, str],
        sample_user_piece: PromptRequestPiece,
        sample_assistant_piece: PromptRequestPiece,
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
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

        assert state.turn_count == 2

    @pytest.mark.asyncio
    async def test_update_conversation_state_exceeds_max_turns_raises_error(
        self,
        attack_identifier: dict[str, str],
        sample_user_piece: PromptRequestPiece,
        sample_assistant_piece: PromptRequestPiece,
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
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
        self,
        attack_identifier: dict[str, str],
        sample_user_piece: PromptRequestPiece,
        sample_assistant_piece: PromptRequestPiece,
        sample_score: Score,
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # First add the conversation to memory
        user_response = PromptRequestResponse(request_pieces=[sample_user_piece])
        assistant_response = PromptRequestResponse(request_pieces=[sample_assistant_piece])

        # Manually add to memory to establish the original_prompt_id
        for piece in user_response.request_pieces:
            piece.conversation_id = conversation_id
        for piece in assistant_response.request_pieces:
            piece.conversation_id = conversation_id

        manager._memory.add_request_response_to_memory(request=user_response)
        manager._memory.add_request_response_to_memory(request=assistant_response)

        # Add score to memory
        sample_score.prompt_request_response_id = str(sample_assistant_piece.original_prompt_id)
        manager._memory.add_scores_to_memory(scores=[sample_score])

        # Create conversation ending with assistant message
        conversation = [
            PromptRequestResponse(request_pieces=[sample_user_piece]),
            PromptRequestResponse(request_pieces=[sample_assistant_piece]),
        ]

        state = await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
        )

        # Should extract scores for last assistant message
        assert len(state.last_assistant_message_scores) == 1
        assert state.last_user_message == sample_user_piece.converted_value

    @pytest.mark.asyncio
    async def test_update_conversation_state_no_scores_for_assistant_message(
        self, attack_identifier: dict[str, str], sample_assistant_piece: PromptRequestPiece
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
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
        self, attack_identifier: dict[str, str], sample_assistant_piece: PromptRequestPiece, sample_score: Score
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Add assistant message to memory and add score
        assistant_response = PromptRequestResponse(request_pieces=[sample_assistant_piece])
        for piece in assistant_response.request_pieces:
            piece.conversation_id = conversation_id
        manager._memory.add_request_response_to_memory(request=assistant_response)

        sample_score.prompt_request_response_id = str(sample_assistant_piece.original_prompt_id)
        manager._memory.add_scores_to_memory(scores=[sample_score])

        # Create conversation with assistant messages only
        conversation = [
            PromptRequestResponse(request_pieces=[sample_assistant_piece]),
            PromptRequestResponse(request_pieces=[sample_assistant_piece]),
        ]

        with pytest.raises(ValueError, match="There must be a user message preceding"):
            await manager.update_conversation_state_async(
                conversation_id=conversation_id, prepended_conversation=conversation, max_turns=5
            )


class TestPrivateMethods:
    """Tests for private helper methods"""

    def test_should_exclude_piece_from_memory_single_turn_mode(self, sample_system_piece: PromptRequestPiece):
        # System pieces should be excluded in both single-turn and multi-turn modes
        # because set_system_prompt() is called on the target, which internally adds them to memory
        assert ConversationManager._should_exclude_piece_from_memory(piece=sample_system_piece, max_turns=None)

    def test_should_exclude_piece_from_memory_multi_turn_system_piece(self, sample_system_piece: PromptRequestPiece):
        # System pieces should be excluded in both single-turn and multi-turn modes
        assert ConversationManager._should_exclude_piece_from_memory(piece=sample_system_piece, max_turns=5)

    def test_should_exclude_piece_from_memory_single_turn_non_system_piece(
        self, sample_user_piece: PromptRequestPiece, sample_assistant_piece: PromptRequestPiece
    ):
        # In single-turn mode, non-system pieces should not be excluded
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_user_piece, max_turns=None)
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_assistant_piece, max_turns=None)

    def test_should_exclude_piece_from_memory_multi_turn_non_system_piece(
        self, sample_user_piece: PromptRequestPiece, sample_assistant_piece: PromptRequestPiece
    ):
        # In multi-turn mode, non-system pieces should not be excluded
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_user_piece, max_turns=5)
        assert not ConversationManager._should_exclude_piece_from_memory(piece=sample_assistant_piece, max_turns=5)


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_empty_request_pieces(self, attack_identifier: dict[str, str]):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create request with empty pieces list
        conversation = [PromptRequestResponse(request_pieces=[])]

        state = await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=conversation
        )

        # Should handle gracefully
        assert state.turn_count == 0
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 0

    @pytest.mark.asyncio
    async def test_update_conversation_state_with_none_request(self, attack_identifier: dict[str, str]):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Create conversation with None values
        conversation = [None]  # This would be caught by type checking in real code

        # Should handle gracefully
        state = await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=conversation  # type: ignore
        )

        assert state.turn_count == 0

    @pytest.mark.asyncio
    async def test_update_conversation_state_preserves_piece_metadata(
        self, attack_identifier: dict[str, str], sample_user_piece: PromptRequestPiece
    ):
        manager = ConversationManager(attack_identifier=attack_identifier)
        conversation_id = str(uuid.uuid4())

        # Add metadata to piece
        sample_user_piece.labels = {"test": "label"}
        sample_user_piece.prompt_metadata = {"key": "value", "count": 1}

        conversation = [PromptRequestResponse(request_pieces=[sample_user_piece])]

        await manager.update_conversation_state_async(
            conversation_id=conversation_id, prepended_conversation=conversation
        )

        # Verify piece was processed with metadata intact
        stored_conversation = manager.get_conversation(conversation_id)
        assert len(stored_conversation) == 1
        processed_piece = stored_conversation[0].request_pieces[0]
        assert processed_piece.labels == {"test": "label"}
        assert processed_piece.prompt_metadata == {"key": "value", "count": 1}

    def test_conversation_state_dataclass_defaults(self):
        # Test ConversationState initialization with defaults
        state = ConversationState()

        assert state.turn_count == 0
        assert state.last_user_message == ""
        assert state.last_assistant_message_scores == []

    def test_conversation_state_dataclass_with_values(self, sample_score: Score):
        # Test ConversationState initialization with custom values
        state = ConversationState(
            turn_count=5, last_user_message="Test message", last_assistant_message_scores=[sample_score]
        )

        assert state.turn_count == 5
        assert state.last_user_message == "Test message"
        assert state.last_assistant_message_scores == [sample_score]
