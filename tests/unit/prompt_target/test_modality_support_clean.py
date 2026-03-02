# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for modality support detection using set[frozenset[PromptDataType]] architecture.

- SUPPORTED_INPUT_MODALITIES is set[frozenset[PromptDataType]]
- Each frozenset represents a valid combination of modalities
- Exact frozenset matching for precise modality detection
"""

from unittest.mock import AsyncMock

import pytest

from pyrit.models import Message, MessagePiece, PromptDataType
from pyrit.prompt_target.modality_verification import (
    _create_test_message,
    verify_target_modalities,
)
from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget
from pyrit.prompt_target.text_target import TextTarget


class TestModalitySupport:
    """Test modality support detection with set[frozenset[PromptDataType]] architecture."""

    def test_text_target_input_modalities(self, patch_central_database):
        """Test TextTarget only supports text input."""
        target = TextTarget()

        assert target.input_modality_supported({"text"})
        assert not target.input_modality_supported({"text", "image_path"})
        assert not target.input_modality_supported({"image_path"})
        assert not target.input_modality_supported({"text", "audio_path"})

    def test_text_target_output_modalities(self, patch_central_database):
        """Test TextTarget only supports text output."""
        target = TextTarget()

        assert target.output_modality_supported({"text"})
        assert not target.output_modality_supported({"image_path"})
        assert not target.output_modality_supported({"text", "image_path"})

        expected_output = {frozenset(["text"])}
        assert target.SUPPORTED_OUTPUT_MODALITIES == expected_output

    def test_openai_static_api_declarations(self, patch_central_database):
        """Test OpenAI uses static API modality declarations, not model-name pattern matching.

        All OpenAI models get the same static API declarations regardless of model name.
        The optional verify_actual_modalities() trims these down at runtime.
        """
        model_names = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "some-future-model-xyz"]

        for model_name in model_names:
            target = OpenAIChatTarget(
                model_name=model_name,
                endpoint="https://mock.azure.com/",
                api_key="mock-api-key",
            )

            expected_api_modalities = {
                frozenset(["text"]),
                frozenset(["text", "image_path"]),
                frozenset(["text", "audio_path"]),
            }
            assert target.SUPPORTED_INPUT_MODALITIES == expected_api_modalities, (
                f"Model {model_name} should declare full API modalities"
            )

            assert target.input_modality_supported({"text"})
            assert target.input_modality_supported({"text", "image_path"})
            assert target.input_modality_supported({"text", "audio_path"})

    def test_openai_unsupported_combinations(self, patch_central_database):
        """Test that OpenAI rejects modality combinations not declared by the API."""
        target = OpenAIChatTarget(
            model_name="gpt-4o",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )

        assert not target.input_modality_supported({"image_path"})
        assert not target.input_modality_supported({"audio_path"})
        assert not target.input_modality_supported({"text", "image_path", "audio_path"})

    def test_frozenset_order_independence(self, patch_central_database):
        """Test that modality checking is order-independent via frozenset matching."""
        target = OpenAIChatTarget(
            model_name="gpt-4o",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )

        assert target.input_modality_supported({"image_path", "text"})
        assert target.input_modality_supported({"text", "image_path"})

    def test_verify_actual_modalities_exists(self, patch_central_database):
        """Test the optional runtime verification method exists."""
        target = TextTarget()
        assert hasattr(target, "verify_actual_modalities")

    def test_modality_type_validation(self, patch_central_database):
        """Test that modality checking works with PromptDataType literals."""
        target = TextTarget()

        text_type: PromptDataType = "text"
        image_type: PromptDataType = "image_path"
        audio_type: PromptDataType = "audio_path"

        assert target.input_modality_supported({text_type})
        assert not target.input_modality_supported({text_type, image_type})
        assert not target.input_modality_supported({audio_type})

    def test_create_test_message_single_modality(self):
        """Test that _create_test_message works for a single text modality."""
        msg = _create_test_message(frozenset(["text"]))
        assert len(msg.message_pieces) == 1
        assert msg.message_pieces[0].original_value_data_type == "text"
        assert msg.message_pieces[0].original_value == "test"

    def test_create_test_message_multimodal(self):
        """Test that _create_test_message creates a valid Message for multimodal inputs.

        All pieces must share the same conversation_id and role for Message.validate() to pass.
        """
        msg = _create_test_message(frozenset(["text", "image_path"]))
        assert len(msg.message_pieces) == 2
        data_types = {p.original_value_data_type for p in msg.message_pieces}
        assert data_types == {"text", "image_path"}

        # Verify all pieces share conversation_id (required by Message.validate)
        conv_ids = {p.conversation_id for p in msg.message_pieces}
        assert len(conv_ids) == 1

    @pytest.mark.asyncio
    async def test_verify_target_modalities_success(self, patch_central_database):
        """Test verify_target_modalities returns supported modalities on success."""
        target = TextTarget()

        # Mock send_prompt_async to return a successful response
        response_piece = MessagePiece(
            role="assistant",
            original_value="ok",
            original_value_data_type="text",
            response_error="none",
        )
        mock_response = Message([response_piece])
        target.send_prompt_async = AsyncMock(return_value=[mock_response])

        result = await verify_target_modalities(target)
        assert frozenset(["text"]) in result

    @pytest.mark.asyncio
    async def test_verify_target_modalities_exception(self, patch_central_database):
        """Test verify_target_modalities excludes modalities that raise exceptions."""
        target = TextTarget()
        target.send_prompt_async = AsyncMock(side_effect=Exception("unsupported modality"))

        result = await verify_target_modalities(target)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_verify_target_modalities_error_response(self, patch_central_database):
        """Test verify_target_modalities excludes modalities returning error responses."""
        target = TextTarget()

        response_piece = MessagePiece(
            role="assistant",
            original_value="content filter triggered",
            original_value_data_type="text",
            response_error="blocked",
        )
        mock_response = Message([response_piece])
        target.send_prompt_async = AsyncMock(return_value=[mock_response])

        result = await verify_target_modalities(target)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_verify_target_modalities_partial_support(self, patch_central_database):
        """Test verify_target_modalities with a target that supports some but not all modalities."""
        target = OpenAIChatTarget(
            model_name="gpt-4o",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )

        # Text succeeds, text+image raises
        async def selective_send(*, message):
            types = {p.original_value_data_type for p in message.message_pieces}
            if "image_path" in types:
                raise Exception("image not supported by this model")
            response_piece = MessagePiece(
                role="assistant",
                original_value="ok",
                original_value_data_type="text",
                response_error="none",
            )
            return [Message([response_piece])]

        target.send_prompt_async = selective_send

        result = await verify_target_modalities(target)
        assert frozenset(["text"]) in result
        assert frozenset(["text", "image_path"]) not in result
