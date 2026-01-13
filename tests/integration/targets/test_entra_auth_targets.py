# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os

import pytest

from pyrit.auth import (
    get_azure_openai_auth,
    get_azure_token_provider,
)
from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_target import (
    OpenAIChatTarget,
    OpenAIImageTarget,
    OpenAIResponseTarget,
    OpenAITTSTarget,
    OpenAIVideoTarget,
    PromptShieldTarget,
    RealtimeTarget,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name", "supports_seed"),
    [
        ("AZURE_OPENAI_GPT4O_ENDPOINT", "AZURE_OPENAI_GPT4O_MODEL", True),
        ("AZURE_OPENAI_GPT4O_ENDPOINT2", "AZURE_OPENAI_GPT4O_MODEL2", True),
        ("AZURE_OPENAI_GPT4O_AAD_ENDPOINT", "AZURE_OPENAI_GPT4O_AAD_MODEL", True),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL", True),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL2", True),
        ("AZURE_OPENAI_INTEGRATION_TEST_ENDPOINT", "AZURE_OPENAI_INTEGRATION_TEST_MODEL", True),
        ("AZURE_OPENAI_GPT4O_STRICT_FILTER_ENDPOINT", "AZURE_OPENAI_GPT4O_STRICT_FILTER_MODEL", True),
        ("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT", "AZURE_OPENAI_GPT3_5_CHAT_MODEL", True),
        ("AZURE_OPENAI_GPT4_CHAT_ENDPOINT", "AZURE_OPENAI_GPT4_CHAT_MODEL", True),
        ("AZURE_OPENAI_GPT5_COMPLETIONS_ENDPOINT", "AZURE_OPENAI_GPT5_COMPLETIONS_MODEL", True),
        ("AZURE_OPENAI_GPTV_CHAT_ENDPOINT", "AZURE_OPENAI_GPTV_CHAT_MODEL", True),
        ("AZURE_FOUNDRY_DEEPSEEK_ENDPOINT", "", True),
        ("AZURE_FOUNDRY_PHI4_ENDPOINT", "", True),
    ],
)
async def test_openai_chat_target_entra_auth(sqlite_instance, endpoint, model_name, supports_seed):
    endpoint_value = os.environ[endpoint]
    args = {
        "endpoint": endpoint_value,
        "temperature": 0.0,
        "api_key": get_azure_openai_auth(endpoint_value),
        "model_name": os.environ[model_name] if model_name else None,
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
    ("endpoint", "model_name"),
    [
        ("OPENAI_IMAGE_ENDPOINT1", "OPENAI_IMAGE_MODEL1"),
        ("OPENAI_IMAGE_ENDPOINT2", "OPENAI_IMAGE_MODEL2"),
    ],
)
async def test_openai_image_target_entra_auth(sqlite_instance, endpoint, model_name):
    endpoint_value = os.environ[endpoint]
    args = {
        "endpoint": endpoint_value,
        "api_key": get_azure_openai_auth(endpoint_value),
        "model_name": os.environ[model_name],
    }

    target = OpenAIImageTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="A cute baby sea otter")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("OPENAI_TTS_ENDPOINT1", "OPENAI_TTS_MODEL1"),
        ("OPENAI_TTS_ENDPOINT2", "OPENAI_TTS_MODEL2"),
    ],
)
async def test_openai_tts_target_entra_auth(sqlite_instance, endpoint, model_name):
    endpoint_value = os.environ[endpoint]
    args = {
        "endpoint": endpoint_value,
        "api_key": get_azure_openai_auth(endpoint_value),
        "model_name": os.environ[model_name],
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
        ("AZURE_OPENAI_GPT41_RESPONSES_ENDPOINT", "AZURE_OPENAI_GPT41_RESPONSES_MODEL"),
        ("AZURE_OPENAI_GPT5_RESPONSES_ENDPOINT", "AZURE_OPENAI_GPT5_MODEL"),
    ],
)
async def test_openai_responses_target_entra_auth(sqlite_instance, endpoint, model_name):
    endpoint_value = os.environ[endpoint]
    args = {
        "endpoint": endpoint_value,
        "model_name": os.environ[model_name],
        "api_key": get_azure_openai_auth(endpoint_value),
    }

    target = OpenAIResponseTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("OPENAI_REALTIME_ENDPOINT", "OPENAI_REALTIME_MODEL"),
    ],
)
async def test_openai_realtime_target_entra_auth(sqlite_instance, endpoint, model_name):
    endpoint_value = os.environ[endpoint]
    model_name_value = os.environ[model_name]
    target = RealtimeTarget(
        endpoint=endpoint_value,
        model_name=model_name_value,
        api_key=get_azure_openai_auth(endpoint_value),
    )

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
async def test_video_target_entra_auth(sqlite_instance):
    # Takes a long time and sometimes encounters retry errors.
    # Note: OPENAI_VIDEO_ENDPOINT should be configured for Sora v2 API
    endpoint = os.environ["OPENAI_VIDEO2_ENDPOINT"]
    target = OpenAIVideoTarget(
        endpoint=endpoint,
        model_name=os.environ["OPENAI_VIDEO2_MODEL"],
        api_key=get_azure_openai_auth(endpoint),
    )
    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="test")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
async def test_prompt_shield_target_entra_auth(sqlite_instance):
    # Make sure to assign the Cognitive Services User or Contributor role
    endpoint = os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"]
    target = PromptShieldTarget(
        endpoint=endpoint,
        api_key=get_azure_token_provider("https://cognitiveservices.azure.com/.default"),
        field="userPrompt",
    )
    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="test")
    assert result is not None
    assert result.last_response is not None
