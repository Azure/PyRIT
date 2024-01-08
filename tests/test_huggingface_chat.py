# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest

from pyrit.chat.hugging_face_chat import HuggingFaceChat
from pyrit.models import ChatMessage


@pytest.fixture
def hf_chat() -> HuggingFaceChat:
    with patch.object(HuggingFaceChat, "load_model_and_tokenizer") as mock_load_model_and_tokenizer:
        hf_online_bot = HuggingFaceChat(model_id="cognitivecomputations/WizardLM-7B-Uncensored")
        mock_load_model_and_tokenizer.assert_called_once()
    return hf_online_bot


def test_initialization_with_parameters(hf_chat: HuggingFaceChat):
    assert hf_chat.model_id == "cognitivecomputations/WizardLM-7B-Uncensored"
    assert hf_chat.use_cuda is False
    assert hf_chat.tokenizer is None
    assert hf_chat.model is None
    assert hf_chat.tensor_format == "pt"


@patch("transformers.PretrainedConfig.from_pretrained")
def test_is_model_id_valid_true(mock_from_pretrained, hf_chat: HuggingFaceChat):
    # Simulate a successful load
    mock_from_pretrained.return_value = MagicMock()
    assert hf_chat.is_model_id_valid() is True


@patch("transformers.PretrainedConfig.from_pretrained")
def test_is_model_id_valid_false(mock_from_pretrained, hf_chat: HuggingFaceChat):
    # Simulate a failure in loading the model
    mock_from_pretrained.side_effect = Exception("Invalid model ID")
    assert hf_chat.is_model_id_valid() is False


def test_complete_chat_failure(hf_chat: HuggingFaceChat):
    # Set up the mock return values for tokenizer and model
    hf_chat.tokenizer = MagicMock()
    hf_chat.model = MagicMock()
    messages = [ChatMessage(role="user", content="Hello")]
    with pytest.raises(ValueError) as e:
        _ = hf_chat.complete_chat(messages)
    assert str(e.value) == "At least two chat message objects are required for the first call. Obtained only 1."


def test_complete_chat_success(hf_chat: HuggingFaceChat):
    # Set up the mock return values for tokenizer and model
    hf_chat.tokenizer = MagicMock()
    hf_chat.model = MagicMock()
    messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content1"),
        ChatMessage(role="assistant", content="assistant1"),
        ChatMessage(role="user", content="user content2"),
    ]
    response = hf_chat.complete_chat(messages)
    assert isinstance(response, MagicMock)


def test_extract_last_assistant_response_multiple_markers(hf_chat: HuggingFaceChat):
    text = "USER: HelloASSISTANT: HiUSER: How are you?ASSISTANT: I'm good, thanks!"
    assert hf_chat.extract_last_assistant_response(text) == " I'm good, thanks!"


def test_extract_last_assistant_response_single_marker(hf_chat: HuggingFaceChat):
    text = "USER: HelloASSISTANT: Hi there"
    assert hf_chat.extract_last_assistant_response(text) == " Hi there"


def test_extract_last_assistant_response_no_marker(hf_chat: HuggingFaceChat):
    text = "USER: Hello, how are you?"
    assert hf_chat.extract_last_assistant_response(text) == ""


def test_extract_last_assistant_response_with_closing_token(hf_chat: HuggingFaceChat):
    text = "USER: HelloASSISTANT: Hi there</s>USER: Bye"
    assert hf_chat.extract_last_assistant_response(text) == "Hi there"


def test_extract_last_assistant_response_without_closing_token(
    hf_chat: HuggingFaceChat,
):
    text = "USER: HelloASSISTANT: Hi there"
    assert hf_chat.extract_last_assistant_response(text) == " Hi there"
