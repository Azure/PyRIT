# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_target import (
    OpenAIChatTarget,
    OpenAIDALLETarget,
    OpenAIResponseTarget,
    OpenAISoraTarget,
    OpenAITTSTarget,
    PromptShieldTarget,
    RealtimeTarget,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("AZURE_OPENAI_GPT4O_ENDPOINT", ""),
        ("AZURE_OPENAI_GPT4O_ENDPOINT2", ""),
        ("AZURE_OPENAI_GPT4O_AAD_ENDPOINT", ""),
        ("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT", ""),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2", ""),
        ("AZURE_OPENAI_INTEGRATION_TEST_ENDPOINT", ""),
        ("AZURE_OPENAI_GPT4O_STRICT_FILTER_ENDPOINT", ""),
        ("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT", ""),
        ("AZURE_OPENAI_GPT4_CHAT_ENDPOINT", ""),
        ("AZURE_OPENAI_GPTV_CHAT_ENDPOINT", ""),
        ("AZURE_FOUNDRY_DEEPSEEK_ENDPOINT", ""),
        ("AZURE_FOUNDRY_PHI4_ENDPOINT", ""),
        ("AZURE_FOUNDRY_MINSTRAL3B_ENDPOINT", ""),
        ("XPIAI_OPENAI_GPT4O_CHAT_ENDPOINT", "XPIA_OPENAI_MODEL"),
    ],
)
async def test_openai_chat_target_entra_auth(sqlite_instance, endpoint, model_name):
    args = {
        "endpoint": os.getenv(endpoint),
        "temperature": 0.0,
        "seed": 42,
        "use_entra_auth": True,
        "model_name": os.getenv(model_name),
    }

    # These endpoints should have Entra authentication enabled in the current context
    # e.g. Cognitive Services OpenAI Contributor or Cognitive Services User/Contributor role (for non-OpenAI resources)
    target = OpenAIChatTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "OPENAI_DALLE_ENDPOINT1",
        "OPENAI_DALLE_ENDPOINT2",
    ],
)
async def test_openai_dalle_target_entra_auth(sqlite_instance, endpoint):
    args = {
        "endpoint": os.getenv(endpoint),
        "use_entra_auth": True,
    }

    target = OpenAIDALLETarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="A cute baby sea otter")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "OPENAI_TTS_ENDPOINT1",
        "OPENAI_TTS_ENDPOINT2",
    ],
)
async def test_openai_tts_target_entra_auth(sqlite_instance, endpoint):
    args = {
        "endpoint": os.getenv(endpoint),
        "use_entra_auth": True,
    }

    target = OpenAITTSTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("OPENAI_RESPONSES_ENDPOINT", "OPENAI_RESPONSES_MODEL"),
    ],
)
async def test_openai_responses_target_entra_auth(sqlite_instance, endpoint, model_name):
    args = {"endpoint": os.getenv(endpoint), "model_name": os.getenv(model_name), "use_entra_auth": True}

    target = OpenAIResponseTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("OPENAI_REALTIME_ENDPOINT", ""),
    ],
)
async def test_openai_realtime_target_entra_auth(sqlite_instance, endpoint, model_name):
    args = {"endpoint": os.getenv(endpoint), "model_name": model_name, "use_entra_auth": True}

    target = RealtimeTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.run_only_if_all_tests
@pytest.mark.asyncio
async def test_sora_target_entra_auth(sqlite_instance):
    # Takes a long time and sometimes encounters retry errors.
    target = OpenAISoraTarget(endpoint=os.getenv("SORA_ENDPOINT"), use_entra_auth=True)
    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="test")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
async def test_prompt_shield_target_entra_auth(sqlite_instance):
    # Make sure to assign the Cognitive Services User or Contributor role
    target = PromptShieldTarget(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
        use_entra_auth=True,
        field="userPrompt",
    )
    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="test")
    assert result is not None
    assert result.last_response is not None
