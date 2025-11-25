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
    ("endpoint", "model_name", "supports_seed"),
    [
        ("AZURE_OPENAI_GPT4O_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPT4O_NEW_FORMAT_ENDPOINT", "AZURE_OPENAI_GPT4O_MODEL", True),
        ("AZURE_OPENAI_GPT4O_ENDPOINT2", "", True),
        ("AZURE_OPENAI_GPT4O_AAD_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2", "", True),
        ("AZURE_OPENAI_INTEGRATION_TEST_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPT4O_STRICT_FILTER_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPT4_CHAT_ENDPOINT", "", True),
        ("AZURE_OPENAI_GPTV_CHAT_ENDPOINT", "", True),
        ("AZURE_FOUNDRY_DEEPSEEK_ENDPOINT", "", True),
        ("AZURE_FOUNDRY_PHI4_ENDPOINT", "", True),
        ("AZURE_FOUNDRY_MINSTRAL3B_ENDPOINT", "", False),
        ("XPIA_OPENAI_GPT4O_ENDPOINT", "XPIA_OPENAI_MODEL", True),
    ],
)
async def test_openai_chat_target_entra_auth(sqlite_instance, endpoint, model_name, supports_seed):
    args = {
        "endpoint": os.getenv(endpoint),
        "temperature": 0.0,
        "use_entra_auth": True,
        "model_name": os.getenv(model_name),
    }

    if supports_seed:
        args["seed"] = 42

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
        ("AZURE_OPENAI_RESPONSES_NEW_FORMAT_ENDPOINT", "AZURE_OPENAI_RESPONSES_MODEL"),
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


@pytest.mark.asyncio
async def test_sora_target_entra_auth(sqlite_instance):
    # Takes a long time and sometimes encounters retry errors.
    # Note: OPENAI_SORA_ENDPOINT should be configured for Sora v2 API
    target = OpenAISoraTarget(
        endpoint=os.getenv("OPENAI_SORA2_ENDPOINT"),
        model_name=os.getenv("OPENAI_SORA2_MODEL"),
        use_entra_auth=True)
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
