# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget, RealtimeTarget


@pytest.mark.asyncio
@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("AZURE_OPENAI_GPT4O_AAD_ENDPOINT", ""),
    ],
)
async def test_openai_chat_target_aad_auth(duckdb_instance, endpoint, model_name):
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

    orchestrator = PromptSendingOrchestrator(objective_target=target)
    result = await orchestrator.send_prompts_async(prompt_list=["Hello, how are you?"])
    assert result is not None
    assert result[0].request_pieces[0].converted_value is not None


@pytest.mark.asyncio
@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize(
    ("endpoint", "model_name"),
    [
        ("OPENAI_REALTIME_ENDPOINT", ""),
    ],
)
async def test_openai_realtime_target_aad_auth(duckdb_instance, endpoint, model_name):
    args = {"endpoint": os.getenv(endpoint), "model_name": model_name}

    # These endpoints should have AAD authentication enabled in the current context
    # e.g.  Cognitive Services OpenAI Contributor role
    target = RealtimeTarget(**args)

    orchestrator = PromptSendingOrchestrator(objective_target=target)
    result = await orchestrator.send_prompts_async(prompt_list=["Hello, how are you?"])
    assert result is not None
    assert result[0].request_pieces[0].converted_value is not None
