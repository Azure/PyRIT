# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_target import OpenAIChatTarget, RealtimeTarget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("AZURE_OPENAI_GPT4O_AAD_ENDPOINT", ""),
    ],
)
async def test_openai_chat_target_aad_auth(sqlite_instance, endpoint, model_name):
    args = {
        "endpoint": os.getenv(endpoint),
        "temperature": 0.0,
        "seed": 42,
        "use_aad_auth": True,
        "model_name": model_name,
    }

    # These endpoints should have AAD authentication enabled in the current context
    # e.g. Cognitive Services OpenAI Contributor role
    target = OpenAIChatTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None


@pytest.mark.asyncio
@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("OPENAI_REALTIME_ENDPOINT", ""),
    ],
)
async def test_openai_realtime_target_aad_auth(sqlite_instance, endpoint, model_name):
    args = {"endpoint": os.getenv(endpoint), "model_name": model_name}

    # These endpoints should have AAD authentication enabled in the current context
    # e.g.  Cognitive Services OpenAI Contributor role
    target = RealtimeTarget(**args)

    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective="Hello, how are you?")
    assert result is not None
    assert result.last_response is not None
