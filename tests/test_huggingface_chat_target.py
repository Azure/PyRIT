# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest
from unittest.mock import patch, MagicMock
from pyrit.prompt_target.hugging_face_chat_target import HuggingFaceChatTarget
from pyrit.models.prompt_request_response import PromptRequestResponse, PromptRequestPiece
import asyncio


# Fixture to mock download_model_with_cli globally for all tests
@pytest.fixture(autouse=True)
def mock_download_model_with_cli():
    with patch("pyrit.prompt_target.hugging_face_chat_target.download_model_with_cli", return_value=None):
        yield


# Fixture to mock download_specific_files_with_cli globally for all tests
@pytest.fixture(autouse=True)
def mock_download_specific_files_with_cli():
    with patch("pyrit.prompt_target.hugging_face_chat_target.download_specific_files_with_cli", return_value=None):
        yield


# Fixture to mock os.path.exists to prevent file system access
@pytest.fixture(autouse=True)
def mock_os_path_exists():
    with patch("os.path.exists", return_value=True):
        yield


# Mock torch.cuda.is_available to prevent CUDA-related errors during testing
@pytest.fixture(autouse=True)
def mock_torch_cuda_is_available():
    with patch("torch.cuda.is_available", return_value=False):
        yield


# Mock the AutoTokenizer and AutoModelForCausalLM to prevent actual model loading
@pytest.fixture(autouse=True)
def mock_transformers():
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()) as mock_tokenizer:
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()) as mock_model:
            yield mock_tokenizer, mock_model


# Mock PretrainedConfig.from_pretrained to prevent actual configuration loading
@pytest.fixture(autouse=True)
def mock_pretrained_config():
    with patch("transformers.PretrainedConfig.from_pretrained", return_value=MagicMock()):
        yield


def test_initialization():
    # Test the initialization without loading the actual models
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    assert hf_chat.model_id == "test_model"
    assert not hf_chat.use_cuda
    assert hf_chat.device == "cpu"
    assert hf_chat.model is not None
    assert hf_chat.tokenizer is not None


def test_is_model_id_valid_true():
    # Simulate valid model ID
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    assert hf_chat.is_model_id_valid()


def test_is_model_id_valid_false():
    # Simulate invalid model ID by causing an exception
    with patch("transformers.PretrainedConfig.from_pretrained", side_effect=Exception("Invalid model")):
        hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
        assert not hf_chat.is_model_id_valid()


def test_load_model_and_tokenizer():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    assert hf_chat.model is not None
    assert hf_chat.tokenizer is not None


def test_send_prompt_success():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)

    # Mock the methods used in send_prompt_async
    hf_chat.tokenizer.chat_template = MagicMock()
    hf_chat.tokenizer.apply_chat_template.return_value = MagicMock()
    hf_chat.model.generate.return_value = [[101, 102, 103]]
    hf_chat.tokenizer.decode.return_value = "Assistant's response"

    request_piece = PromptRequestPiece(
        role="user",
        original_value="Hello, how are you?",
        converted_value="Hello, how are you?",
        converted_value_data_type="text",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Run the asynchronous method
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(hf_chat.send_prompt_async(prompt_request=prompt_request))

    # Access the response text via request_pieces
    assert response.request_pieces[0].original_value == "Assistant's response"


def test_missing_chat_template_error():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    hf_chat.tokenizer.chat_template = None

    request_piece = PromptRequestPiece(
        role="user",
        original_value="Hello, how are you?",
        converted_value="Hello, how are you?",
        converted_value_data_type="text",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError) as excinfo:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hf_chat.send_prompt_async(prompt_request=prompt_request))

    assert "Tokenizer does not have a chat template" in str(excinfo.value)


def test_invalid_prompt_request_validation():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)

    # Create an invalid prompt request with multiple request pieces
    request_piece1 = PromptRequestPiece(
        role="user", original_value="First piece", converted_value="First piece", converted_value_data_type="text"
    )
    request_piece2 = PromptRequestPiece(
        role="user", original_value="Second piece", converted_value="Second piece", converted_value_data_type="text"
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece1, request_piece2])

    with pytest.raises(ValueError) as excinfo:
        hf_chat._validate_request(prompt_request=prompt_request)

    assert "This target only supports a single prompt request piece." in str(excinfo.value)


def test_load_with_missing_files():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False, necessary_files=["file1", "file2"])
    assert hf_chat.model is not None
    assert hf_chat.tokenizer is not None


def test_enable_disable_cache():
    # Test enabling cache
    HuggingFaceChatTarget.enable_cache()
    assert HuggingFaceChatTarget._cache_enabled

    # Test disabling cache
    HuggingFaceChatTarget.disable_cache()
    assert not HuggingFaceChatTarget._cache_enabled
    assert HuggingFaceChatTarget._cached_model is None
    assert HuggingFaceChatTarget._cached_tokenizer is None
    assert HuggingFaceChatTarget._cached_model_id is None
