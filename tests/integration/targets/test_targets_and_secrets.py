# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import (
    AzureMLChatTarget,
    OpenAIChatTarget,
    OpenAICompletionTarget,
    OpenAIDALLETarget,
    OpenAITTSTarget,
    RealtimeTarget,
)


async def _assert_can_send_prompt(target, check_if_llm_interpreted_request=True):
    simple_prompt = "simply repeat this word back to me. Don't say anything else. Say: 'test'"
    orchestrator = PromptSendingOrchestrator(objective_target=target)

    all_prompts = [simple_prompt]

    result = await orchestrator.send_prompts_async(prompt_list=all_prompts)
    assert len(result) == 1
    assert len(result[0].request_pieces) == 1
    if check_if_llm_interpreted_request:
        assert "test" in result[0].request_pieces[0].converted_value.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name", "no_api_version"),
    [
        ("PLATFORM_OPENAI_CHAT_ENDPOINT", "PLATFORM_OPENAI_CHAT_KEY", "PLATFORM_OPENAI_CHAT_MODEL", False),
        ("AZURE_OPENAI_GPT4O_ENDPOINT", "AZURE_OPENAI_GPT4O_KEY", "", False),
        ("AZURE_OPENAI_GPT4O_INTEGRATION_TEST_ENDPOINT", "AZURE_OPENAI_GPT4O_INTEGRATION_TEST_KEY", "", False),
        ("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY", "", False),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2", "", False),
        ("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT", "AZURE_OPENAI_GPT3_5_CHAT_KEY", "", False),
        ("AZURE_OPENAI_GPT4_CHAT_ENDPOINT", "AZURE_OPENAI_GPT4_CHAT_KEY", "", False),
        ("AZURE_OPENAI_GPTV_CHAT_ENDPOINT", "AZURE_OPENAI_GPTV_CHAT_KEY", "", False),
        ("AZURE_FOUNDRY_DEEPSEEK_ENDPOINT", "AZURE_FOUNDRY_DEEPSEEK_KEY", "", False),
        ("AZURE_FOUNDRY_PHI4_ENDPOINT", "AZURE_CHAT_PHI4_KEY", "", False),
        ("AZURE_FOUNDRY_MINSTRAL3B_ENDPOINT", "AZURE_CHAT_MINSTRAL3B_KEY", "", False),
        ("GOOGLE_GEMINI_ENDPOINT", "GOOGLE_GEMINI_API_KEY", "GOOGLE_GEMINI_MODEL", True),
    ],
)
async def test_connect_required_openai_text_targets(duckdb_instance, endpoint, api_key, model_name, no_api_version):
    args = {
        "endpoint": os.getenv(endpoint),
        "api_key": os.getenv(api_key),
        "model_name": os.getenv(model_name),
    }
    if no_api_version:
        args["api_version"] = None
    target = OpenAIChatTarget(**args)

    await _assert_can_send_prompt(target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("AZURE_OPENAI_REALTIME_ENDPOINT", "AZURE_OPENAI_REALTIME_API_KEY", "AZURE_OPENAI_REALTIME_MODEL"),
    ],
)
async def test_connect_required_realtime_targets(duckdb_instance, endpoint, api_key, model_name):
    target = RealtimeTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model_name),
    )

    simple_prompt = "simply repeat this word back to me. Don't say anything else. Say: 'test'"
    orchestrator = PromptSendingOrchestrator(objective_target=target)

    all_prompts = [simple_prompt]

    result = await orchestrator.send_prompts_async(prompt_list=all_prompts)
    assert len(result) == 1
    assert len(result[0].request_pieces) == 2
    assert result[0].request_pieces[0].converted_value_data_type == "text"
    assert result[0].request_pieces[1].converted_value_data_type == "audio_path"
    assert "test" in result[0].request_pieces[0].converted_value.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("AZURE_ML_MANAGED_ENDPOINT", "AZURE_ML_KEY"),
        ("AZURE_ML_MIXTRAL_ENDPOINT", "AZURE_ML_MIXTRAL_KEY"),
    ],
)
async def test_connect_required_aml_text_targets(duckdb_instance, endpoint, api_key):
    target = AzureMLChatTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target)


@pytest.mark.asyncio
async def test_connect_openai_completion(duckdb_instance):

    endpoint = "OPENAI_COMPLETION_ENDPOINT"
    api_key = "OPENAI_COMPLETION_API_KEY"
    model = "OPENAI_COMPLETION_MODEL"

    target = OpenAICompletionTarget(
        endpoint=os.getenv(endpoint), api_key=os.getenv(api_key), model_name=os.getenv(model)
    )

    await _assert_can_send_prompt(target, check_if_llm_interpreted_request=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("OPENAI_DALLE_ENDPOINT1", "OPENAI_DALLE_API_KEY1"),
        ("OPENAI_DALLE_ENDPOINT2", "OPENAI_DALLE_API_KEY2"),
    ],
)
async def test_connect_dall_e(duckdb_instance, endpoint, api_key):
    target = OpenAIDALLETarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target, check_if_llm_interpreted_request=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("OPENAI_TTS_ENDPOINT1", "OPENAI_TTS_KEY1"),
        ("OPENAI_TTS_ENDPOINT2", "OPENAI_TTS_KEY2"),
    ],
)
async def test_connect_tts(duckdb_instance, endpoint, api_key):
    target = OpenAITTSTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target, check_if_llm_interpreted_request=False)


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
        ("OLLAMA_CHAT_ENDPOINT", "", "OLLAMA_MODEL"),
    ],
)
async def test_connect_non_required_openai_text_targets(duckdb_instance, endpoint, api_key, model_name):
    target = OpenAIChatTarget(
        endpoint=os.getenv(endpoint), api_key=os.getenv(api_key), model_name=os.getenv(model_name)
    )

    await _assert_can_send_prompt(target)
