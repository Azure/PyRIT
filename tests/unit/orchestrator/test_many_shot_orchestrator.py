# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.datasets import fetch_many_shot_jailbreaking_dataset
from pyrit.orchestrator import ManyShotJailbreakOrchestrator
from pyrit.prompt_target import PromptChatTarget


@pytest.fixture
def mock_objective_target(patch_central_database) -> MockPromptTarget:
    return MockPromptTarget()


@pytest.fixture
def many_shot_jailbreak_orchestrator_static_examples(mock_objective_target):
    return ManyShotJailbreakOrchestrator(objective_target=mock_objective_target, num_examples=3, verbose=True)


@pytest.fixture
def many_shot_jailbreak_orchestrator_random_examples(mock_objective_target):
    return ManyShotJailbreakOrchestrator(objective_target=mock_objective_target, num_examples=3, verbose=True)


@pytest.mark.asyncio
async def test_construct_many_shot_dialogue_static_examples(many_shot_jailbreak_orchestrator_static_examples):
    prompt = ["How do I pick a lock?"]
    examples = fetch_many_shot_jailbreaking_dataset()[:3]
    with patch("random.sample") as mock_samples:
        mock_samples.return_value = fetch_many_shot_jailbreaking_dataset()[:3]
        dialogue = await many_shot_jailbreak_orchestrator_static_examples._construct_many_shot_dialogue(prompt)
        dialogue = str(dialogue)
        for example in examples:
            assert example["user"] in dialogue
            assert example["assistant"] in dialogue
        assert prompt[0] in dialogue


@pytest.mark.asyncio
async def test_send_single_prompt_static_examples(
    many_shot_jailbreak_orchestrator_static_examples, mock_objective_target
):
    with patch("random.sample") as mock_samples:
        mock_samples.return_value = fetch_many_shot_jailbreaking_dataset()[:3]

        prompt = ["How do I pick a lock?"]
        expected_prompt = await many_shot_jailbreak_orchestrator_static_examples._construct_many_shot_dialogue(
            prompt[0]
        )

        await many_shot_jailbreak_orchestrator_static_examples.send_prompts_async(prompt_list=prompt)
        assert mock_objective_target.prompt_sent == [expected_prompt]


@pytest.mark.asyncio
async def test_send_single_prompt_random_samples(
    many_shot_jailbreak_orchestrator_random_examples, mock_objective_target
):
    prompt = ["How do I pick a lock?"]
    sent_prompt = prompt[0]

    await many_shot_jailbreak_orchestrator_random_examples.send_prompts_async(prompt_list=prompt)
    assert len(mock_objective_target.prompt_sent[0]) > len(sent_prompt)


def test_init_static_examples(many_shot_jailbreak_orchestrator_static_examples):
    assert isinstance(many_shot_jailbreak_orchestrator_static_examples._objective_target, PromptChatTarget)
    assert many_shot_jailbreak_orchestrator_static_examples.num_examples == 3
    assert many_shot_jailbreak_orchestrator_static_examples._verbose is True


def test_init_random_examples(many_shot_jailbreak_orchestrator_random_examples):
    assert isinstance(many_shot_jailbreak_orchestrator_random_examples._objective_target, PromptChatTarget)
    assert many_shot_jailbreak_orchestrator_random_examples.num_examples == 3
    assert many_shot_jailbreak_orchestrator_random_examples._verbose is True
