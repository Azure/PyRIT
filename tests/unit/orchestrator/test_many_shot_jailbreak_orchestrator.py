# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.orchestrator import ManyShotJailbreakOrchestrator


@pytest.fixture
def mock_objective_target(patch_central_database) -> MockPromptTarget:
    return MockPromptTarget()


@pytest.fixture
def many_shot_examples():
    return [
        {"user": "question1", "assistant": "answer1"},
        {"user": "question2", "assistant": "answer2"},
        {"user": "question3", "assistant": "answer3"},
    ]


@pytest.mark.parametrize("n_prompts", [1, 2, 3, 100])
@pytest.mark.parametrize("explicit_examples", [True, False])
@pytest.mark.parametrize("example_count", [1, 3, 100])
@pytest.mark.asyncio
async def test_many_shot_orchestrator(
    explicit_examples, n_prompts, example_count, many_shot_examples, mock_objective_target
):
    with patch(
        "pyrit.orchestrator.single_turn.many_shot_jailbreak_orchestrator.fetch_many_shot_jailbreaking_dataset"
    ) as mock_fetch:
        mock_fetch.return_value = many_shot_examples
        if explicit_examples:
            examples = many_shot_examples
        else:
            examples = None

        prompts = [f"prompt{i}" for i in range(n_prompts)]
        orchestrator = ManyShotJailbreakOrchestrator(
            objective_target=mock_objective_target,
            many_shot_examples=examples,
            example_count=example_count,
        )
        assert (
            len(orchestrator._examples) == len(many_shot_examples)
            if len(many_shot_examples) <= example_count
            else example_count
        )

        await orchestrator.send_prompts_async(prompt_list=prompts)
        assert len(mock_objective_target.prompt_sent) == n_prompts
        for i in range(n_prompts):
            assert f"prompt{i}" in mock_objective_target.prompt_sent[i]
            for example_dict in orchestrator._examples:
                for role, content in example_dict.items():
                    assert role in mock_objective_target.prompt_sent[i].lower()
                    assert content in mock_objective_target.prompt_sent[i]

        if explicit_examples:
            assert mock_fetch.call_count == 0
        else:
            mock_fetch.assert_called_once()


@pytest.mark.asyncio
async def test_send_prompts_async_empty_prompt_list(many_shot_examples, mock_objective_target):
    orchestrator = ManyShotJailbreakOrchestrator(objective_target=mock_objective_target)
    with pytest.raises(ValueError, match="Prompt list must not be empty."):
        await orchestrator.send_prompts_async(prompt_list=[])
