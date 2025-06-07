# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.models import SeedPrompt
from pyrit.orchestrator.multi_turn.tree_of_attacks_node import TreeOfAttacksNode
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score.scorer import Scorer


@pytest.fixture
def tree_of_attack_node(patch_central_database):
    return TreeOfAttacksNode(
        objective_target=MagicMock(spec=PromptTarget),
        adversarial_chat=MagicMock(spec=PromptChatTarget),
        adversarial_chat_seed_prompt=MagicMock(spec=SeedPrompt),
        adversarial_chat_prompt_template=MagicMock(spec=SeedPrompt),
        adversarial_chat_system_seed_prompt=MagicMock(spec=SeedPrompt),
        objective_scorer=MagicMock(spec=Scorer),
        on_topic_scorer=MagicMock(spec=Scorer),
        prompt_converters=[MagicMock(spec=PromptConverter)],
        desired_response_prefix="Hello, I am a prefix",
        orchestrator_id="test_orchestrator_id",
        memory_labels={"label_key": "label_value"},
    )


@pytest.mark.asyncio
async def test_send_prompt_async(tree_of_attack_node):
    tree_of_attack_node._generate_red_teaming_prompt_async = AsyncMock(return_value="generated_prompt")
    tree_of_attack_node._on_topic_scorer.score_text_async = AsyncMock(
        return_value=[MagicMock(get_value=MagicMock(return_value=True))]
    )
    tree_of_attack_node._prompt_normalizer.send_prompt_async = AsyncMock(
        return_value=MagicMock(request_pieces=[MagicMock(id="response_id")])
    )
    tree_of_attack_node._objective_scorer.score_async = AsyncMock(
        return_value=[MagicMock(get_value=MagicMock(return_value=0.9))]
    )

    await tree_of_attack_node.send_prompt_async(objective="test_objective")

    assert tree_of_attack_node.prompt_sent is True
    assert tree_of_attack_node.completed is True
    assert tree_of_attack_node.objective_score.get_value() == 0.9


def test_duplicate(tree_of_attack_node):
    duplicate_node = tree_of_attack_node.duplicate()

    assert duplicate_node._desired_response_prefix == tree_of_attack_node._desired_response_prefix

    assert duplicate_node.parent_id == tree_of_attack_node.node_id
    assert duplicate_node.objective_target_conversation_id != tree_of_attack_node.objective_target_conversation_id
    assert duplicate_node.adversarial_chat_conversation_id != tree_of_attack_node.adversarial_chat_conversation_id


@pytest.mark.asyncio
async def test_generate_red_teaming_prompt_async(tree_of_attack_node):
    with patch.object(tree_of_attack_node, "_memory", create=True) as mock_memory:
        # Set the `get_conversation` method's return value
        mock_memory.get_conversation.return_value = []
        tree_of_attack_node._adversarial_chat_seed_prompt.render_template_value = MagicMock(return_value="seed_prompt")
        tree_of_attack_node._adversarial_chat_system_seed_prompt.render_template_value = MagicMock(
            return_value="system_prompt"
        )

        tree_of_attack_node._prompt_normalizer.send_prompt_async = AsyncMock(
            return_value=MagicMock(get_value=lambda: '{"prompt": "generated_prompt"}')
        )

        prompt = await tree_of_attack_node._generate_red_teaming_prompt_async(objective="test_objective")

        assert prompt == "generated_prompt"
        tree_of_attack_node._adversarial_chat.set_system_prompt.assert_called_once()
        tree_of_attack_node._adversarial_chat_system_seed_prompt.render_template_value.assert_called_once()
        _, named_args = tree_of_attack_node._adversarial_chat_system_seed_prompt.render_template_value.call_args

        assert named_args["desired_prefix"] == tree_of_attack_node._desired_response_prefix


def test_parse_red_teaming_response(tree_of_attack_node):
    response = '{"prompt": "parsed_prompt", "improvement": "some_improvement"}'
    prompt = tree_of_attack_node._parse_red_teaming_response(response)

    assert prompt == "parsed_prompt"
