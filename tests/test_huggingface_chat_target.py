# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch
import pytest
from pyrit.prompt_target.hugging_face_chat_target import HuggingFaceChatTarget
from pyrit.models import ChatMessage


@pytest.fixture
def hf_chat() -> HuggingFaceChatTarget:
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
    ):
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = mock_model_instance
        hf_chat_bot = HuggingFaceChatTarget(model_id="cognitivecomputations/WizardLM-7B-Uncensored")
        mock_tokenizer.assert_called_once_with("cognitivecomputations/WizardLM-7B-Uncensored")
        mock_model.assert_called_once_with("cognitivecomputations/WizardLM-7B-Uncensored")
    return hf_chat_bot


def test_initialization_with_parameters(hf_chat: HuggingFaceChatTarget):
    assert hf_chat.model_id == "cognitivecomputations/WizardLM-7B-Uncensored"
    assert hf_chat.use_cuda is False
    assert isinstance(hf_chat.tokenizer, MagicMock)
    assert isinstance(hf_chat.model, MagicMock)
    assert hf_chat.tensor_format == "pt"


@patch("transformers.PretrainedConfig.from_pretrained")
def test_is_model_id_valid_true(mock_from_pretrained, hf_chat: HuggingFaceChatTarget):
    # Simulate a successful load
    mock_from_pretrained.return_value = MagicMock()
    assert hf_chat.is_model_id_valid() is True


@patch("transformers.PretrainedConfig.from_pretrained")
def test_is_model_id_valid_false(mock_from_pretrained, hf_chat: HuggingFaceChatTarget):
    # Simulate a failure in loading the model
    mock_from_pretrained.side_effect = Exception("Invalid model ID")
    assert hf_chat.is_model_id_valid() is False


def test_complete_chat_failure(hf_chat: HuggingFaceChatTarget):
    # Set up the mock return values for tokenizer and model
    hf_chat.tokenizer = MagicMock()
    hf_chat.model = MagicMock()
    messages = [ChatMessage(role="user", content="Hello")]
    with pytest.raises(ValueError) as e:
        _ = hf_chat.complete_chat(messages)
    assert str(e.value) == "At least two chat message objects are required for the first call. Obtained only 1."


def test_complete_chat_success(hf_chat: HuggingFaceChatTarget):
    # Set up the mock return values for tokenizer and model
    hf_chat.tokenizer = MagicMock()
    hf_chat.model = MagicMock()
    hf_chat.model.generate.return_value = [[101, 102, 103]]  # Mocked output tokens
    hf_chat.tokenizer.decode.return_value = "Generated response"

    # Mock the extract_last_assistant_response method to return the expected response
    hf_chat.extract_last_assistant_response = MagicMock(return_value="Generated response")

    messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content1"),
        ChatMessage(role="assistant", content="assistant1"),
        ChatMessage(role="user", content="user content2"),
    ]

    # Call the method and get the response
    response = hf_chat.complete_chat(messages)

    # Assert that the response matches the mocked return value
    assert response == "Generated response"


def test_extract_last_assistant_response_multiple_markers(hf_chat: HuggingFaceChatTarget):
    text = "USER: HelloASSISTANT: HiUSER: How are you?ASSISTANT: I'm good, thanks!"
    assert hf_chat.extract_last_assistant_response(text) == "I'm good, thanks!"  # Removed leading space


def test_extract_last_assistant_response_single_marker(hf_chat: HuggingFaceChatTarget):
    text = "USER: HelloASSISTANT: Hi there"
    assert hf_chat.extract_last_assistant_response(text) == "Hi there"  # Removed leading space


def test_extract_last_assistant_response_no_marker(hf_chat: HuggingFaceChatTarget):
    text = "USER: Hello, how are you?"
    assert hf_chat.extract_last_assistant_response(text) == ""


def test_extract_last_assistant_response_with_closing_token(hf_chat: HuggingFaceChatTarget):
    text = "USER: HelloASSISTANT: Hi there</s>USER: Bye"
    assert hf_chat.extract_last_assistant_response(text) == "Hi there"


def test_extract_last_assistant_response_without_closing_token(hf_chat: HuggingFaceChatTarget):
    text = "USER: HelloASSISTANT: Hi there"
    assert hf_chat.extract_last_assistant_response(text) == "Hi there"  # Removed leading space
