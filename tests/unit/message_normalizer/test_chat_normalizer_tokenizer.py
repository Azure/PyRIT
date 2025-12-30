# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import textwrap

import pytest
from transformers import AutoTokenizer
from unittest.mock import MagicMock, patch

from pyrit.message_normalizer import TokenizerTemplateNormalizer
from pyrit.models import Message, MessagePiece


def _make_message(role: str, content: str) -> Message:
    """Helper to create a Message from role and content."""
    return Message(message_pieces=[MessagePiece(role=role, original_value=content)])


class TestTokenizerTemplateNormalizerInit:
    """Tests for TokenizerTemplateNormalizer initialization."""

    def test_init_with_tokenizer(self):
        """Test direct initialization with a tokenizer."""
        mock_tokenizer = MagicMock()
        normalizer = TokenizerTemplateNormalizer(tokenizer=mock_tokenizer)
        assert normalizer.tokenizer == mock_tokenizer
        assert normalizer.system_message_behavior == "keep"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        mock_tokenizer = MagicMock()
        normalizer = TokenizerTemplateNormalizer(
            tokenizer=mock_tokenizer,
            system_message_behavior="developer",
        )
        assert normalizer.system_message_behavior == "developer"

    def test_model_aliases_contains_expected_aliases(self):
        """Test that MODEL_ALIASES contains expected aliases."""
        aliases = TokenizerTemplateNormalizer.MODEL_ALIASES
        assert "chatml" in aliases
        assert "phi3" in aliases
        assert "qwen" in aliases
        assert "llama3" in aliases


class TestFromModel:
    """Tests for the from_model factory method."""

    def test_from_model_with_alias(self):
        """Test from_model resolves alias to full model name."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_tokenizer.chat_template = "some template"
            mock_from_pretrained.return_value = mock_tokenizer

            normalizer = TokenizerTemplateNormalizer.from_model("chatml")

            mock_from_pretrained.assert_called_once_with(
                "HuggingFaceH4/zephyr-7b-beta", token=None
            )
            assert normalizer.tokenizer == mock_tokenizer

    def test_from_model_with_full_name(self):
        """Test from_model works with full model name."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_tokenizer.chat_template = "some template"
            mock_from_pretrained.return_value = mock_tokenizer

            normalizer = TokenizerTemplateNormalizer.from_model("custom/model-name")

            mock_from_pretrained.assert_called_once_with("custom/model-name", token=None)
            assert normalizer.tokenizer == mock_tokenizer

    def test_from_model_with_token(self):
        """Test from_model passes token for gated models."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_tokenizer.chat_template = "some template"
            mock_from_pretrained.return_value = mock_tokenizer

            TokenizerTemplateNormalizer.from_model("some-model", token="hf_token123")

            mock_from_pretrained.assert_called_once_with("some-model", token="hf_token123")

    def test_from_model_raises_when_no_chat_template(self):
        """Test from_model raises ValueError if tokenizer has no chat_template."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_tokenizer.chat_template = None
            mock_from_pretrained.return_value = mock_tokenizer

            with pytest.raises(ValueError, match="does not have a chat_template"):
                TokenizerTemplateNormalizer.from_model("model-without-template")

    def test_from_model_case_insensitive_alias(self):
        """Test from_model aliases are case-insensitive."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_tokenizer.chat_template = "some template"
            mock_from_pretrained.return_value = mock_tokenizer

            TokenizerTemplateNormalizer.from_model("CHATML")

            mock_from_pretrained.assert_called_once_with(
                "HuggingFaceH4/zephyr-7b-beta", token=None
            )


class TestNormalizeStringAsync:
    """Tests for the normalize_string_async method."""

    @pytest.fixture
    def chatml_tokenizer_normalizer(self):
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        return TokenizerTemplateNormalizer(tokenizer=tokenizer)

    @pytest.mark.asyncio
    async def test_normalize_chatml(self, chatml_tokenizer_normalizer: TokenizerTemplateNormalizer):
        messages = [
            _make_message("system", "You are a friendly chatbot who always responds in the style of a pirate"),
            _make_message("user", "How many helicopters can a human eat in one sitting?"),
        ]
        expected = textwrap.dedent(
            """\
            <|system|>
            You are a friendly chatbot who always responds in the style of a pirate</s>
            <|user|>
            How many helicopters can a human eat in one sitting?</s>
            <|assistant|>
            """
        )

        assert await chatml_tokenizer_normalizer.normalize_string_async(messages) == expected

    @pytest.mark.asyncio
    async def test_normalize_uses_converted_value(self):
        """Test that normalize uses converted_value when available."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        normalizer = TokenizerTemplateNormalizer(tokenizer=mock_tokenizer)

        piece = MessagePiece(
            role="user",
            original_value="original",
            converted_value="converted",
        )
        messages = [Message(message_pieces=[piece])]

        await normalizer.normalize_string_async(messages)

        call_args = mock_tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == [{"role": "user", "content": "converted"}]

    @pytest.mark.asyncio
    async def test_normalize_falls_back_to_original_value(self):
        """Test that normalize falls back to original_value when converted_value is None."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        normalizer = TokenizerTemplateNormalizer(tokenizer=mock_tokenizer)

        piece = MessagePiece(
            role="user",
            original_value="original",
            converted_value=None,
        )
        messages = [Message(message_pieces=[piece])]

        await normalizer.normalize_string_async(messages)

        call_args = mock_tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == [{"role": "user", "content": "original"}]
