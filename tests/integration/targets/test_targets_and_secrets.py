# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
import pytest

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget

from pyrit.common import path
from pyrit.prompt_target.azure_ml_chat_target import AzureMLChatTarget


async def _assert_can_send_prompt(target):
    simple_prompt = "simply repeat this word back to me. Don't say anything else. Say: 'test'"
    orchestrator = PromptSendingOrchestrator(objective_target=target)

    all_prompts = [simple_prompt]

    result = await orchestrator.send_prompts_async(prompt_list=all_prompts)
    assert len(result) == 1
    assert len(result[0].request_pieces) == 1
    assert "test" in result[0].request_pieces[0].converted_value.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("OPENAI_CHAT_ENDPOINT", "OPENAI_CHAT_KEY", "OPENAI_CHAT_MODEL"),
        ("AZURE_OPENAI_GPT4O_ENDPOINT", "AZURE_OPENAI_GPT4O_KEY", ""),
        ("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT", "AZURE_OPENAI_GPT3_5_CHAT_KEY",""),
        ("AZURE_OPENAI_GPT4_CHAT_ENDPOINT", "AZURE_OPENAI_GPT4_CHAT_KEY", ""),
        ("AZURE_OPENAI_GPTV_CHAT_ENDPOINT", "AZURE_OPENAI_GPTV_CHAT_KEY", ""),
        ("AZURE_FOUNDRY_DEEPSEEK_ENDPOINT", "AZURE_FOUNDRY_DEEPSEEK_KEY", ""),
        ("AZURE_FOUNDRY_PHI4_ENDPOINT", "AZURE_CHAT_PHI4_KEY", ""),
        ("AZURE_FOUNDRY_MINSTRAL3B_ENDPOINT", "AZURE_CHAT_MINSTRAL3B_KEY", ""),
    ]
)
async def test_connect_required_openai_text_targets(endpoint, api_key, model_name):

    initialize_pyrit(memory_db_type=IN_MEMORY)

    target = OpenAIChatTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model_name)
    )

    await _assert_can_send_prompt(target)


# These tests are not run in our pipeline because they just use personal secrets
# but still useful to check the endpoints work
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("PLATFORM_OPENAI_ENDPOINT", "PLATFORM_OPENAI_KEY", "PLATFORM_OPENAI_GPT4O_MODEL"),
        ("GROQ_ENDPOINT", "GROQ_KEY", "GROQ_LLAMA_MODEL"),
        ("OPEN_ROUTER_ENDPOINT", "OPEN_ROUTER_KEY", "OPEN_ROUTER_CLAUDE_MODEL"),
    ]
)
async def test_connect_non_required_openai_text_targets(endpoint, api_key, model_name):

    initialize_pyrit(memory_db_type=IN_MEMORY)

    if os.getenv("RUN_ALL_TESTS").lower() != "true":
        pytest.skip("Skipping test because RUN_ALL_TESTS is not set to true")

    target = OpenAIChatTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model_name)
    )

    await _assert_can_send_prompt(target)



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("AZURE_ML_MANAGED_ENDPOINT", "AZURE_ML_KEY"),
        ("AZURE_ML_MIXTRAL_ENDPOINT", "AZURE_ML_MIXTRAL_KEY"),
    ]
)
async def test_connect_required_aml_text_targets(endpoint, api_key):

    initialize_pyrit(memory_db_type=IN_MEMORY)

    target = AzureMLChatTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target)