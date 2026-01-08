# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackConverterConfig, RTASystemPromptPaths
from pyrit.executor.attack.multi_turn.simulated_conversation import (
    SimulatedConversationResult,
    SimulatedTargetSystemPromptPaths,
    generate_simulated_conversation_async,
)
from pyrit.models import AttackOutcome, AttackResult, Message, MessagePiece, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import TrueFalseScorer


@pytest.fixture
def mock_adversarial_chat() -> MagicMock:
    """Create a mock adversarial chat target for testing."""
    chat = MagicMock(spec=PromptChatTarget)
    chat.send_prompt_async = AsyncMock()
    chat.set_system_prompt = MagicMock()
    chat.get_identifier.return_value = {"__type__": "MockAdversarialChat", "__module__": "test_module"}
    return chat


@pytest.fixture
def mock_objective_scorer() -> MagicMock:
    """Create a mock objective scorer for testing."""
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.score_async = AsyncMock()
    scorer.get_identifier.return_value = {"__type__": "MockScorer", "__module__": "test_module"}
    return scorer


@pytest.fixture
def adversarial_system_prompt_path() -> Path:
    """Return a valid adversarial chat system prompt path for testing."""
    return RTASystemPromptPaths.TEXT_GENERATION.value


@pytest.fixture
def sample_conversation() -> list[Message]:
    """Create a sample conversation for testing."""
    return [
        Message(
            message_pieces=[
                MessagePiece(
                    role="user",
                    original_value="Hello",
                    original_value_data_type="text",
                    conversation_id=str(uuid.uuid4()),
                )
            ]
        ),
        Message(
            message_pieces=[
                MessagePiece(
                    role="assistant",
                    original_value="Hi there!",
                    original_value_data_type="text",
                    conversation_id=str(uuid.uuid4()),
                )
            ]
        ),
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestSimulatedTargetSystemPromptPaths:
    """Tests for SimulatedTargetSystemPromptPaths enum."""

    def test_compliant_path_exists(self):
        """Test that the COMPLIANT system prompt path points to an existing file."""
        path = SimulatedTargetSystemPromptPaths.COMPLIANT.value
        assert isinstance(path, Path)
        assert path.exists(), f"Expected compliant.yaml at {path}"

    def test_compliant_path_is_yaml(self):
        """Test that the COMPLIANT path is a YAML file."""
        path = SimulatedTargetSystemPromptPaths.COMPLIANT.value
        assert path.suffix == ".yaml"


@pytest.mark.usefixtures("patch_central_database")
class TestGenerateSimulatedConversationAsync:
    """Tests for generate_simulated_conversation_async function."""

    @pytest.mark.asyncio
    async def test_raises_error_for_zero_turns(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
    ):
        """Test that zero num_turns raises ValueError."""
        with pytest.raises(ValueError, match="num_turns must be a positive integer"):
            await generate_simulated_conversation_async(
                objective="Test objective",
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
                adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                num_turns=0,
            )

    @pytest.mark.asyncio
    async def test_raises_error_for_negative_turns(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
    ):
        """Test that negative num_turns raises ValueError."""
        with pytest.raises(ValueError, match="num_turns must be a positive integer"):
            await generate_simulated_conversation_async(
                objective="Test objective",
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
                adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                num_turns=-1,
            )

    @pytest.mark.asyncio
    async def test_uses_adversarial_chat_as_simulated_target(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that the same adversarial_chat is used as simulated target."""
        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            # Configure the mock attack
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                )

                # Verify RedTeamingAttack was created with adversarial_chat as objective_target
                mock_attack_class.assert_called_once()
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["objective_target"] == mock_adversarial_chat

    @pytest.mark.asyncio
    async def test_creates_attack_with_score_last_turn_only_true(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that the attack is created with score_last_turn_only=True."""
        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                )

                # Verify score_last_turn_only was set to True
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["score_last_turn_only"] is True

    @pytest.mark.asyncio
    async def test_creates_attack_with_correct_max_turns(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that the attack is created with the specified num_turns as max_turns."""
        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=5,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=5,
                )

                # Verify max_turns was set correctly
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["max_turns"] == 5

    @pytest.mark.asyncio
    async def test_returns_simulated_conversation_result(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that the function returns a SimulatedConversationResult."""
        conversation_id = str(uuid.uuid4())
        mock_score = MagicMock(spec=Score)

        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=conversation_id,
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                    last_score=mock_score,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                result = await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                )

                # Verify get_conversation was called with the correct conversation_id
                mock_memory.get_conversation.assert_called_once_with(conversation_id=conversation_id)

                # Verify the result is a SimulatedConversationResult
                assert isinstance(result, SimulatedConversationResult)
                assert result.conversation == sample_conversation
                assert result.score == mock_score

    @pytest.mark.asyncio
    async def test_passes_system_prompt_via_prepended_conversation(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that the simulated target system prompt is passed via prepended_conversation."""
        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                )

                # Verify execute_async was called with a system message in prepended_conversation
                mock_attack.execute_async.assert_called_once()
                call_kwargs = mock_attack.execute_async.call_args.kwargs
                assert "prepended_conversation" in call_kwargs
                prepended = call_kwargs["prepended_conversation"]
                assert len(prepended) == 1
                assert prepended[0].role == "system"

    @pytest.mark.asyncio
    async def test_passes_memory_labels_to_execute(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that memory_labels are passed to attack.execute_async."""
        memory_labels = {"source": "test", "type": "simulated"}

        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                    memory_labels=memory_labels,
                )

                # Verify memory_labels were passed to execute_async
                execute_kwargs = mock_attack.execute_async.call_args.kwargs
                assert execute_kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_passes_converter_config_to_attack(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that attack_converter_config is passed to RedTeamingAttack."""
        converter_config = AttackConverterConfig()

        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                    attack_converter_config=converter_config,
                )

                # Verify converter_config was passed to RedTeamingAttack
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["attack_converter_config"] == converter_config

    @pytest.mark.asyncio
    async def test_prepends_system_message_to_conversation(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that a system message is prepended when executing the attack."""
        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                    num_turns=3,
                )

                # Verify prepended_conversation was passed to execute_async
                execute_kwargs = mock_attack.execute_async.call_args.kwargs
                assert "prepended_conversation" in execute_kwargs
                prepended = execute_kwargs["prepended_conversation"]
                assert len(prepended) == 1
                assert prepended[0].message_pieces[0].role == "system"

    @pytest.mark.asyncio
    async def test_uses_default_num_turns_of_3(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        adversarial_system_prompt_path: Path,
        sample_conversation: list[Message],
    ):
        """Test that default num_turns is 3."""
        with patch("pyrit.executor.attack.multi_turn.simulated_conversation.RedTeamingAttack") as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=str(uuid.uuid4()),
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch("pyrit.executor.attack.multi_turn.simulated_conversation.CentralMemory") as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                # Call without specifying num_turns
                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    adversarial_chat_system_prompt_path=adversarial_system_prompt_path,
                )

                # Verify default max_turns is 3
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["max_turns"] == 3


@pytest.mark.usefixtures("patch_central_database")
class TestSimulatedConversationResult:
    """Tests for SimulatedConversationResult dataclass."""

    def _create_message(self, role: str, content: str) -> Message:
        """Helper to create a Message with the given role and content."""
        return Message(
            message_pieces=[
                MessagePiece(
                    role=role,  # type: ignore[arg-type]
                    original_value=content,
                    original_value_data_type="text",
                    conversation_id=str(uuid.uuid4()),
                )
            ]
        )

    def _create_conversation(self, num_turns: int) -> list[Message]:
        """Helper to create a conversation with the specified number of turns."""
        messages = []
        for i in range(1, num_turns + 1):
            messages.append(self._create_message("user", f"Turn {i} user"))
            messages.append(self._create_message("assistant", f"Turn {i} assistant"))
        return messages

    def test_prepended_messages_default_excludes_last_turn(self):
        """Test prepended_messages excludes last turn when turn_index is None (default)."""
        messages = self._create_conversation(3)  # 6 messages
        result = SimulatedConversationResult(conversation=messages, score=None)

        prepended = result.prepended_messages
        # Should exclude last turn (2 messages)
        assert len(prepended) == 4
        assert prepended[-1].role == "assistant"
        assert prepended[-1].get_value() == "Turn 2 assistant"

    def test_prepended_messages_with_turn_index_2(self):
        """Test prepended_messages with turn_index=2 returns only turn 1."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None, turn_index=2)

        prepended = result.prepended_messages
        # Should return only turn 1 (2 messages)
        assert len(prepended) == 2
        assert prepended[0].get_value() == "Turn 1 user"
        assert prepended[1].get_value() == "Turn 1 assistant"

    def test_prepended_messages_with_turn_index_1_returns_empty(self):
        """Test prepended_messages with turn_index=1 returns empty list."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None, turn_index=1)

        prepended = result.prepended_messages
        assert len(prepended) == 0

    def test_prepended_messages_with_empty_conversation(self):
        """Test prepended_messages returns empty list for empty conversation."""
        result = SimulatedConversationResult(conversation=[], score=None)
        assert result.prepended_messages == []

    def test_prepended_messages_with_single_turn(self):
        """Test prepended_messages returns empty when only one turn."""
        messages = self._create_conversation(1)
        result = SimulatedConversationResult(conversation=messages, score=None)

        prepended = result.prepended_messages
        assert len(prepended) == 0

    def test_next_message_default_returns_last_user(self):
        """Test next_message returns last turn's user message when turn_index is None."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None)

        next_msg = result.next_message
        assert next_msg is not None
        assert next_msg.role == "user"
        assert next_msg.get_value() == "Turn 3 user"

    def test_next_message_with_turn_index_2(self):
        """Test next_message with turn_index=2 returns turn 2's user message."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None, turn_index=2)

        next_msg = result.next_message
        assert next_msg is not None
        assert next_msg.role == "user"
        assert next_msg.get_value() == "Turn 2 user"

    def test_next_message_with_turn_index_1(self):
        """Test next_message with turn_index=1 returns first user message."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None, turn_index=1)

        next_msg = result.next_message
        assert next_msg is not None
        assert next_msg.get_value() == "Turn 1 user"

    def test_next_message_with_empty_conversation(self):
        """Test next_message returns None for empty conversation."""
        result = SimulatedConversationResult(conversation=[], score=None)
        assert result.next_message is None

    def test_turn_index_can_be_set_after_creation(self):
        """Test that turn_index can be modified after creation."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None)

        # Default: last turn (3)
        assert result.next_message is not None
        assert result.next_message.get_value() == "Turn 3 user"
        assert len(result.prepended_messages) == 4

        # Set to turn 2
        result.turn_index = 2
        assert result.next_message is not None
        assert result.next_message.get_value() == "Turn 2 user"
        assert len(result.prepended_messages) == 2

        # Set to turn 1
        result.turn_index = 1
        assert result.next_message is not None
        assert result.next_message.get_value() == "Turn 1 user"
        assert len(result.prepended_messages) == 0

    def test_turn_index_bounded_by_conversation_length(self):
        """Test that turn_index is bounded by available turns."""
        messages = self._create_conversation(2)  # Only 2 turns
        result = SimulatedConversationResult(conversation=messages, score=None, turn_index=10)

        # Should clamp to last available turn (2)
        next_msg = result.next_message
        assert next_msg is not None
        assert next_msg.get_value() == "Turn 2 user"

    def test_turn_index_minimum_is_1(self):
        """Test that turn_index minimum is 1."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None, turn_index=0)

        # Should treat as turn 1
        assert result.next_message is not None
        assert result.next_message.get_value() == "Turn 1 user"
        assert len(result.prepended_messages) == 0

    def test_conversation_with_trailing_user_message(self):
        """Test handling conversation that ends with user message (incomplete turn)."""
        messages = self._create_conversation(2)
        messages.append(self._create_message("user", "Turn 3 user"))  # No assistant response
        result = SimulatedConversationResult(conversation=messages, score=None)

        # The trailing user message should be treated as a turn
        next_msg = result.next_message
        assert next_msg is not None
        assert next_msg.get_value() == "Turn 3 user"
        # prepended should have turns 1 and 2
        assert len(result.prepended_messages) == 4

    def test_conversation_property_returns_full_list(self):
        """Test that conversation property returns the full message list."""
        messages = self._create_conversation(2)
        result = SimulatedConversationResult(conversation=messages, score=None)

        assert result.conversation == messages
        assert len(result.conversation) == 4

    def test_score_property(self):
        """Test that score property returns the stored score."""
        mock_score = MagicMock(spec=Score)
        result = SimulatedConversationResult(conversation=[], score=mock_score)
        assert result.score == mock_score

    def test_score_property_none(self):
        """Test that score property can be None."""
        result = SimulatedConversationResult(conversation=[], score=None)
        assert result.score is None

    def test_prepended_messages_returns_new_ids(self):
        """Test that prepended_messages returns messages with new IDs to avoid database conflicts."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None)

        prepended = result.prepended_messages

        # Verify we got messages with different IDs than the originals
        for i, prepended_msg in enumerate(prepended):
            original_msg = messages[i]
            # IDs should be different
            for orig_piece, new_piece in zip(original_msg.message_pieces, prepended_msg.message_pieces):
                assert new_piece.id != orig_piece.id, "prepended_messages should return messages with new IDs"
            # But content should be the same
            assert prepended_msg.get_value() == original_msg.get_value()

    def test_next_message_returns_new_id(self):
        """Test that next_message returns a message with a new ID to avoid database conflicts."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None)

        next_msg = result.next_message
        assert next_msg is not None

        # The original is at index 4 (turn 3 user message)
        original_msg = messages[4]

        # IDs should be different
        for orig_piece, new_piece in zip(original_msg.message_pieces, next_msg.message_pieces):
            assert new_piece.id != orig_piece.id, "next_message should return a message with a new ID"
        # But content should be the same
        assert next_msg.get_value() == original_msg.get_value()

    def test_prepended_messages_called_twice_returns_different_ids(self):
        """Test that calling prepended_messages twice returns different IDs each time."""
        messages = self._create_conversation(3)
        result = SimulatedConversationResult(conversation=messages, score=None)

        prepended1 = result.prepended_messages
        prepended2 = result.prepended_messages

        # Each call should generate new IDs
        for msg1, msg2 in zip(prepended1, prepended2):
            for piece1, piece2 in zip(msg1.message_pieces, msg2.message_pieces):
                assert piece1.id != piece2.id, "Each call to prepended_messages should return new IDs"
