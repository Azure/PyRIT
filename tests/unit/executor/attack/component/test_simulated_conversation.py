# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack.component.simulated_conversation import (
    SimulatedTargetSystemPromptPaths,
    generate_simulated_conversation_async,
)
from pyrit.executor.attack import AttackConverterConfig
from pyrit.models import AttackOutcome, AttackResult, Message, MessagePiece
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
    ):
        """Test that zero num_turns raises ValueError."""
        with pytest.raises(ValueError, match="num_turns must be a positive integer"):
            await generate_simulated_conversation_async(
                objective="Test objective",
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
                num_turns=0,
            )

    @pytest.mark.asyncio
    async def test_raises_error_for_negative_turns(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
    ):
        """Test that negative num_turns raises ValueError."""
        with pytest.raises(ValueError, match="num_turns must be a positive integer"):
            await generate_simulated_conversation_async(
                objective="Test objective",
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
                num_turns=-1,
            )

    @pytest.mark.asyncio
    async def test_uses_adversarial_chat_as_simulated_target(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        sample_conversation: list[Message],
    ):
        """Test that the same adversarial_chat is used as simulated target."""
        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
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
        sample_conversation: list[Message],
    ):
        """Test that the attack is created with score_last_turn_only=True."""
        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
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
        sample_conversation: list[Message],
    ):
        """Test that the attack is created with the specified num_turns as max_turns."""
        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    num_turns=5,
                )

                # Verify max_turns was set correctly
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["max_turns"] == 5

    @pytest.mark.asyncio
    async def test_returns_conversation_from_memory(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        sample_conversation: list[Message],
    ):
        """Test that the function returns the conversation from memory."""
        conversation_id = str(uuid.uuid4())

        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
            mock_attack = MagicMock()
            mock_attack.get_identifier.return_value = {"__type__": "RedTeamingAttack", "__module__": "pyrit"}
            mock_attack.execute_async = AsyncMock(
                return_value=AttackResult(
                    attack_identifier={"__type__": "RedTeamingAttack"},
                    conversation_id=conversation_id,
                    objective="Test objective",
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=3,
                )
            )
            mock_attack_class.return_value = mock_attack

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                result = await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    num_turns=3,
                )

                # Verify get_conversation was called with the correct conversation_id
                mock_memory.get_conversation.assert_called_once_with(conversation_id=conversation_id)

                # Verify the result matches the sample conversation
                assert result == sample_conversation

    @pytest.mark.asyncio
    async def test_sets_system_prompt_on_simulated_target(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        sample_conversation: list[Message],
    ):
        """Test that set_system_prompt is called on the simulated target (adversarial_chat)."""
        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                    num_turns=3,
                )

                # Verify set_system_prompt was called on the simulated target (adversarial_chat)
                mock_adversarial_chat.set_system_prompt.assert_called_once()
                call_kwargs = mock_adversarial_chat.set_system_prompt.call_args.kwargs
                assert "system_prompt" in call_kwargs
                assert call_kwargs["attack_identifier"] is not None

    @pytest.mark.asyncio
    async def test_passes_memory_labels_to_execute(
        self,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        sample_conversation: list[Message],
    ):
        """Test that memory_labels are passed to attack.execute_async."""
        memory_labels = {"source": "test", "type": "simulated"}

        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
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
        sample_conversation: list[Message],
    ):
        """Test that attack_converter_config is passed to RedTeamingAttack."""
        converter_config = AttackConverterConfig()

        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
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
        sample_conversation: list[Message],
    ):
        """Test that a system message is prepended when executing the attack."""
        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
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
        sample_conversation: list[Message],
    ):
        """Test that default num_turns is 3."""
        with patch(
            "pyrit.executor.attack.component.simulated_conversation.RedTeamingAttack"
        ) as mock_attack_class:
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

            with patch(
                "pyrit.executor.attack.component.simulated_conversation.CentralMemory"
            ) as mock_memory_class:
                mock_memory = MagicMock()
                mock_memory.get_conversation.return_value = iter(sample_conversation)
                mock_memory_class.get_memory_instance.return_value = mock_memory

                # Call without specifying num_turns
                await generate_simulated_conversation_async(
                    objective="Test objective",
                    adversarial_chat=mock_adversarial_chat,
                    objective_scorer=mock_objective_scorer,
                )

                # Verify default max_turns is 3
                call_kwargs = mock_attack_class.call_args.kwargs
                assert call_kwargs["max_turns"] == 3
