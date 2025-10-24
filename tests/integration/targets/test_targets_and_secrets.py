# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path

import pytest

from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_target import (
    AzureMLChatTarget,
    OpenAIChatTarget,
    OpenAICompletionTarget,
    OpenAIDALLETarget,
    OpenAIResponseTarget,
    OpenAISoraTarget,
    OpenAITTSTarget,
    RealtimeTarget,
)


async def _assert_can_send_prompt(target, check_if_llm_interpreted_request=True, max_retries=2):

    simple_prompt = """\
You are a large language model. Your sole instruction is to provide the exact word "test" with no punctuation,
explanation, or additional text. Output only the word "test" and nothing else.
"""

    attack = PromptSendingAttack(objective_target=target)

    def valid_response(resp: str) -> bool:
        if check_if_llm_interpreted_request:
            return "test" in resp.strip().lower()
        else:
            return True

    attempt = 0
    while attempt < max_retries:
        result = await attack.execute_async(objective=simple_prompt)

        if result.last_response:
            response = result.last_response.converted_value
        else:
            response = ""

        if valid_response(str(response)):
            return response

        attempt += 1

    raise AssertionError(f"LLM did not return exactly 'test' after {max_retries} attempts.")


async def _assert_can_send_video_prompt(target):
    """Helper function to test video generation targets."""
    video_prompt = "A raccoon sailing a pirate ship"
    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective=video_prompt)

    # For video generation, verify we got a successful response
    assert result.last_response is not None
    assert result.last_response.converted_value is not None
    assert (
        result.last_response.response_error == "none"
    ), f"Expected successful response, got error: {result.last_response.response_error}"

    # Validate we got a valid video file path
    video_path = Path(result.last_response.converted_value)
    assert video_path.exists(), f"Video file not found at path: {video_path}"
    assert video_path.is_file(), f"Path exists but is not a file: {video_path}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name", "no_api_version", "supports_seed"),
    [
        ("OPENAI_CHAT_ENDPOINT", "OPENAI_CHAT_KEY", "OPENAI_CHAT_MODEL", False, True),
        ("PLATFORM_OPENAI_CHAT_ENDPOINT", "PLATFORM_OPENAI_CHAT_KEY", "PLATFORM_OPENAI_CHAT_MODEL", False, True),
        ("AZURE_OPENAI_GPT4O_ENDPOINT", "AZURE_OPENAI_GPT4O_KEY", "", False, True),
        ("AZURE_OPENAI_INTEGRATION_TEST_ENDPOINT", "AZURE_OPENAI_INTEGRATION_TEST_KEY", "", False, True),
        ("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY", "", False, True),
        ("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2", "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2", "", False, True),
        ("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT", "AZURE_OPENAI_GPT3_5_CHAT_KEY", "", False, True),
        ("AZURE_OPENAI_GPT4_CHAT_ENDPOINT", "AZURE_OPENAI_GPT4_CHAT_KEY", "", False, True),
        ("AZURE_OPENAI_GPTV_CHAT_ENDPOINT", "AZURE_OPENAI_GPTV_CHAT_KEY", "", False, True),
        ("AZURE_FOUNDRY_DEEPSEEK_ENDPOINT", "AZURE_FOUNDRY_DEEPSEEK_KEY", "", False, True),
        ("AZURE_FOUNDRY_PHI4_ENDPOINT", "AZURE_CHAT_PHI4_KEY", "", False, True),
        ("AZURE_FOUNDRY_MINSTRAL3B_ENDPOINT", "AZURE_CHAT_MINSTRAL3B_KEY", "", False, False),
        ("GOOGLE_GEMINI_ENDPOINT", "GOOGLE_GEMINI_API_KEY", "GOOGLE_GEMINI_MODEL", True, False),
        ("ANTHROPIC_CHAT_ENDPOINT", "ANTHROPIC_CHAT_KEY", "ANTHROPIC_CHAT_MODEL", True, False),
    ],
)
async def test_connect_required_openai_text_targets(
    sqlite_instance, endpoint, api_key, model_name, no_api_version, supports_seed
):
    args = {
        "endpoint": os.getenv(endpoint),
        "api_key": os.getenv(api_key),
        "model_name": os.getenv(model_name),
        "temperature": 0.0,
    }
    if no_api_version:
        args["api_version"] = None

    if supports_seed:
        args["seed"] = 42

    target = OpenAIChatTarget(**args)

    await _assert_can_send_prompt(target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name", "no_api_version"),
    [
        (
            "PLATFORM_OPENAI_RESPONSES_ENDPOINT",
            "PLATFORM_OPENAI_RESPONSES_KEY",
            "PLATFORM_OPENAI_RESPONSES_MODEL",
            True,
        ),
        ("AZURE_OPENAI_RESPONSES_ENDPOINT", "AZURE_OPENAI_RESPONSES_KEY", "AZURE_OPENAI_RESPONSES_MODEL", False),
    ],
)
async def test_connect_required_openai_response_targets(sqlite_instance, endpoint, api_key, model_name, no_api_version):
    args = {
        "endpoint": os.getenv(endpoint),
        "api_key": os.getenv(api_key),
        "model_name": os.getenv(model_name),
    }
    if no_api_version:
        args["api_version"] = None

    target = OpenAIResponseTarget(**args)

    # OpenAIResponseTarget returns structured responses (reasoning JSON), so we just need to verify
    # we can send a prompt and get a response, not that it contains specific text
    await _assert_can_send_prompt(target, check_if_llm_interpreted_request=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("AZURE_OPENAI_REALTIME_ENDPOINT", "AZURE_OPENAI_REALTIME_API_KEY", "AZURE_OPENAI_REALTIME_MODEL"),
    ],
)
async def test_connect_required_realtime_targets(sqlite_instance, endpoint, api_key, model_name):
    target = RealtimeTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model_name),
    )

    await _assert_can_send_prompt(target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key"),
    [
        ("AZURE_ML_MANAGED_ENDPOINT", "AZURE_ML_KEY"),
        ("AZURE_ML_MIXTRAL_ENDPOINT", "AZURE_ML_MIXTRAL_KEY"),
    ],
)
async def test_connect_required_aml_text_targets(sqlite_instance, endpoint, api_key):
    target = AzureMLChatTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target)


@pytest.mark.asyncio
async def test_connect_openai_completion(sqlite_instance):

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
async def test_connect_dall_e(sqlite_instance, endpoint, api_key):
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
async def test_connect_tts(sqlite_instance, endpoint, api_key):
    target = OpenAITTSTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
    )

    await _assert_can_send_prompt(target, check_if_llm_interpreted_request=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("OPENAI_SORA1_ENDPOINT", "OPENAI_SORA1_KEY", "OPENAI_SORA1_MODEL"),
        ("OPENAI_SORA2_ENDPOINT", "OPENAI_SORA2_KEY", "OPENAI_SORA2_MODEL"),
        # OpenAI Platform Sora returns HTTP 401 "Missing scopes: api.videos.write" for all requests
        # ("PLATFORM_OPENAI_SORA_ENDPOINT", "PLATFORM_OPENAI_SORA_KEY", "PLATFORM_OPENAI_SORA_MODEL"),
    ],
)
async def test_connect_sora(sqlite_instance, endpoint, api_key, model_name):
    """Test OpenAISoraTarget with both Sora-1 and Sora-2 APIs (auto-detected via endpoint)."""
    target = OpenAISoraTarget(
        endpoint=os.getenv(endpoint),
        api_key=os.getenv(api_key),
        model_name=os.getenv(model_name),
        resolution_dimensions="1280x720",  # Supported by both v1 and v2
        n_seconds=4,  # Supported by both v1 (up to 20s) and v2 (4, 8, or 12s)
    )

    await _assert_can_send_video_prompt(target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_var", "api_key_var", "model_var", "resolution"),
    [
        # Sora-1 - test unsupported resolution (should return processing error)
        ("OPENAI_SORA1_ENDPOINT", "OPENAI_SORA1_KEY", "OPENAI_SORA1_MODEL", "640x360"),
        # Azure OpenAI Sora-2 - test unsupported resolution (should return processing error)
        ("OPENAI_SORA2_ENDPOINT", "OPENAI_SORA2_KEY", "OPENAI_SORA2_MODEL", "640x360"),
        # OpenAI Platform Sora - 1024x1792 returns HTTP 401 "Missing scopes: api.videos.write"
        # ("PLATFORM_OPENAI_SORA_ENDPOINT", "PLATFORM_OPENAI_SORA_KEY", "PLATFORM_OPENAI_SORA_MODEL", "1024x1792"),
    ],
)
async def test_connect_sora_unsupported_resolution_returns_processing_error(
    sqlite_instance, endpoint_var, api_key_var, model_var, resolution
):
    """
    Test that unsupported resolutions return proper processing errors.

    Sora-1: Tests with 640x360 which is not in the supported resolution list.
    Sora-2: The API spec claims support for 1024x1792 and 1792x1024, but these are
            not supported on Azure OpenAI Sora-2 (only on Sora-2-Pro).

    This test verifies that such errors are properly categorized as error="processing"
    rather than crashing or returning unknown error types.
    """
    target = OpenAISoraTarget(
        endpoint=os.getenv(endpoint_var),
        api_key=os.getenv(api_key_var),
        model_name=os.getenv(model_var),
        resolution_dimensions=resolution,
        n_seconds=4,  # Supported by v2 (4, 8, or 12s)
    )

    video_prompt = "A raccoon sailing a pirate ship"
    attack = PromptSendingAttack(objective_target=target)
    result = await attack.execute_async(objective=video_prompt)

    # Verify we got a response
    assert result.last_response is not None
    assert result.last_response.converted_value is not None

    # This resolution should return a processing error (not supported on this account)
    assert (
        result.last_response.response_error == "processing"
    ), f"Expected error type 'processing' for unsupported resolution, got: {result.last_response.response_error}"
    assert (
        result.last_response.original_value_data_type == "error"
    ), f"Expected response type 'error', got: {result.last_response.original_value_data_type}"

    # Verify the error message contains expected information about invalid resolution
    error_message = result.last_response.converted_value
    # Both v1 and v2 APIs return errors mentioning the problem
    assert any(
        keyword in error_message.lower()
        for keyword in ["user error", "video_generation_user_error", "invalid", "unsupported", "not supported"]
    ), f"Expected error message about invalid/unsupported resolution, got: {error_message}"


##################################################
# Optional tests - not run in pipeline, only locally
# Need RUN_ALL_TESTS=true environment variable to run
###################################################


@pytest.mark.run_only_if_all_tests
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        ("GROQ_ENDPOINT", "GROQ_KEY", "GROQ_LLAMA_MODEL"),
        ("OPEN_ROUTER_ENDPOINT", "OPEN_ROUTER_KEY", "OPEN_ROUTER_CLAUDE_MODEL"),
        ("OLLAMA_CHAT_ENDPOINT", "", "OLLAMA_MODEL"),
    ],
)
async def test_connect_non_required_openai_text_targets(sqlite_instance, endpoint, api_key, model_name):
    target = OpenAIChatTarget(
        endpoint=os.getenv(endpoint), api_key=os.getenv(api_key), model_name=os.getenv(model_name)
    )

    await _assert_can_send_prompt(target)
