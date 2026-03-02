# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Integration tests for modality verification across popular targets.

These tests call verify_target_modalities() against real endpoints to confirm
that modality detection works end-to-end. Each target kind is represented at
least once.
"""

import os

import pytest

from pyrit.prompt_target import (
    OpenAIChatTarget,
    OpenAIImageTarget,
    OpenAIResponseTarget,
    OpenAITTSTarget,
    OpenAIVideoTarget,
    TextTarget,
)
from pyrit.prompt_target.modality_verification import verify_target_modalities


def _get_required_env_var(env_var_name: str) -> str:
    value = os.getenv(env_var_name)
    if not value:
        raise ValueError(f"Environment variable {env_var_name} is not set.")
    return value


# ---------------------------------------------------------------------------
# TextTarget – no credentials needed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_text_target(sqlite_instance):
    """TextTarget supports text-only. Verification should confirm this without any API call."""
    target = TextTarget()

    result = await verify_target_modalities(target)
    # TextTarget.send_prompt_async writes to a stream, so the text modality should succeed
    assert frozenset(["text"]) in result


# ---------------------------------------------------------------------------
# OpenAI Chat – vision-capable model (e.g. gpt-4o)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_chat_vision(sqlite_instance):
    """A vision-capable OpenAI model should support text and text+image."""
    endpoint = _get_required_env_var("AZURE_OPENAI_GPT4O_ENDPOINT")
    api_key = _get_required_env_var("AZURE_OPENAI_GPT4O_KEY")
    model_name = _get_required_env_var("AZURE_OPENAI_GPT4O_MODEL")

    target = OpenAIChatTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result
    assert frozenset(["text", "image_path"]) in result


# ---------------------------------------------------------------------------
# OpenAI Chat – text-only model (e.g. gpt-3.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_chat_text_only(sqlite_instance):
    """A text-only OpenAI model may still accept image input without error (ignoring it).

    Verification detects modalities that the API *rejects*, not what the model
    truly understands.  GPT-3.5 accepts images silently, so we only assert
    that text is confirmed supported.
    """
    endpoint = os.getenv("AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_GPT3_5_CHAT_KEY")
    model_name = os.getenv("AZURE_OPENAI_GPT3_5_CHAT_MODEL")

    if not endpoint or not api_key or not model_name:
        pytest.skip("GPT-3.5 env vars not set")

    target = OpenAIChatTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result


# ---------------------------------------------------------------------------
# OpenAI Chat – negative case: gpt-4 with text+audio should fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_chat_no_audio(sqlite_instance):
    """GPT-4 does not support audio input. Verification should exclude text+audio."""
    endpoint = os.getenv("AZURE_OPENAI_GPT4_CHAT_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_GPT4_CHAT_KEY")
    model_name = os.getenv("AZURE_OPENAI_GPT4_CHAT_MODEL")

    if not endpoint or not api_key or not model_name:
        pytest.skip("GPT-4 env vars not set")

    target = OpenAIChatTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result
    assert frozenset(["text", "audio_path"]) not in result


# ---------------------------------------------------------------------------
# OpenAI Response API – GPT-5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_response_gpt5(sqlite_instance):
    """GPT-5 on the Responses API should support text and text+image."""
    endpoint = os.getenv("AZURE_OPENAI_GPT5_RESPONSES_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_GPT5_KEY")
    model_name = os.getenv("AZURE_OPENAI_GPT5_MODEL")

    if not endpoint or not api_key or not model_name:
        pytest.skip("GPT-5 Responses env vars not set")

    target = OpenAIResponseTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result
    assert frozenset(["text", "image_path"]) in result


# ---------------------------------------------------------------------------
# OpenAI Image API – gpt-image
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_image(sqlite_instance):
    """Image target should support text (generation) and text+image (editing)."""
    endpoint = os.getenv("OPENAI_IMAGE_ENDPOINT2")
    api_key = os.getenv("OPENAI_IMAGE_API_KEY2")
    model_name = os.getenv("OPENAI_IMAGE_MODEL2")

    if not endpoint or not api_key or not model_name:
        pytest.skip("Image API env vars not set")

    target = OpenAIImageTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result


# ---------------------------------------------------------------------------
# OpenAI Video API – Sora
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_video_sora(sqlite_instance):
    """Sora video target should support text-to-video."""
    endpoint = os.getenv("AZURE_OPENAI_VIDEO_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_VIDEO_KEY")
    model_name = os.getenv("AZURE_OPENAI_VIDEO_MODEL")

    if not endpoint or not api_key or not model_name:
        pytest.skip("Video/Sora env vars not set")

    target = OpenAIVideoTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result


# ---------------------------------------------------------------------------
# OpenAI TTS – text input only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_modalities_openai_tts(sqlite_instance):
    """TTS target accepts text input. Verification should confirm text is supported."""
    endpoint = os.getenv("OPENAI_TTS_ENDPOINT")
    api_key = os.getenv("OPENAI_TTS_KEY")
    model_name = os.getenv("OPENAI_TTS_MODEL")

    if not endpoint or not api_key or not model_name:
        pytest.skip("TTS env vars not set")

    target = OpenAITTSTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name,
        voice="alloy",
        response_format="wav",
    )

    result = await verify_target_modalities(target)
    assert frozenset(["text"]) in result
