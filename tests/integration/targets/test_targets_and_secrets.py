# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget

from pyrit.prompt_target.azure_ml_chat_target import AzureMLChatTarget
from pyrit.prompt_target.openai.openai_completion_target import OpenAICompletionTarget
from pyrit.prompt_target.openai.openai_dall_e_target import OpenAIDALLETarget
from pyrit.prompt_target.openai.openai_tts_target import OpenAITTSTarget


async def _assert_can_send_prompt(target, verify_response_text=True):
    simple_prompt = "simply repeat this word back to me. Don't say anything else. Say: 'test'"
    orchestrator = PromptSendingOrchestrator(objective_target=target)

    all_prompts = [simple_prompt]

    result = await orchestrator.send_prompts_async(prompt_list=all_prompts)
    assert len(result) == 1
    assert len(result[0].request_pieces) == 1
    if verify_response_text:
        assert "test" in result[0].request_pieces[0].converted_value.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("AZURE_OPENAI_GPT4O_ENDPOINT", "AZURE_OPENAI_GPT4O_KEY", ""),
        ("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY", ""),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2", ""),
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

@pytest.mark.asyncio
async def test_connect_openai_completion():

    endpoint = "OPENAI_COMPLETEION_ENDPOINT"
    api_key="OPENAI_COMPLETION_API_KEY"
    model="OPENAI_COMPLETION_MODEL"

    initialize_pyrit(memory_db_type=IN_MEMORY)

    target = OpenAICompletionTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model)
    )

    await _assert_can_send_prompt(target, verify_response_text=False)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("OPENAI_DALLE_ENDPOINT1", "OPENAI_DALLE_API_KEY1"),
        ("OPENAI_DALLE_ENDPOINT2", "OPENAI_DALLE_API_KEY2"),
    ]
)
async def test_connect_dall_e(endpoint, api_key):

    initialize_pyrit(memory_db_type=IN_MEMORY)

    target = OpenAIDALLETarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target, verify_response_text=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("OPENAI_TTS_ENDPOINT1", "OPENAI_TTS_KEY1"),
        ("OPENAI_TTS_ENDPOINT2", "OPENAI_TTS_KEY2"),
    ]
)
async def test_connect_tts(endpoint, api_key):

    initialize_pyrit(memory_db_type=IN_MEMORY)

    target = OpenAITTSTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target, verify_response_text=False)



##################################################
# Optional tests - not run in pipeline, only locally
# Need RUN_ALL_TESTS=true environment variable to run
###################################################


@pytest.mark.run_only_if_all_tests
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

    target = OpenAIChatTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model_name)
    )

    await _assert_can_send_prompt(target)