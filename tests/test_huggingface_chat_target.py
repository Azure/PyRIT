# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest
from unittest.mock import patch, MagicMock
from pyrit.prompt_target.hugging_face_chat_target import HuggingFaceChatTarget
from pyrit.models.prompt_request_response import PromptRequestResponse, PromptRequestPiece
import asyncio


# Fixture to mock download_specific_files_with_aria2 globally for all tests
@pytest.fixture(autouse=True)
def mock_download_specific_files_with_aria2():
    with patch(
        "pyrit.common.download_hf_model_with_aria2.download_specific_files_with_aria2", return_value=None
    ) as mock_download:
        yield mock_download


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
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_from_pretrained:
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = MagicMock()
        tokenized_chat_mock = MagicMock()
        tokenized_chat_mock.to.return_value = tokenized_chat_mock

        mock_tokenizer.apply_chat_template.return_value = tokenized_chat_mock
        mock_tokenizer.decode.return_value = "Assistant's response"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_from_pretrained:
            mock_model = MagicMock()
            mock_model.generate.return_value = [[101, 102, 103]]
            mock_model_from_pretrained.return_value = mock_model

            yield mock_tokenizer_from_pretrained, mock_model_from_pretrained


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


@pytest.mark.asyncio
async def test_send_prompt_async():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)

    request_piece = PromptRequestPiece(
        role="user",
        original_value="Hello, how are you?",
        converted_value="Hello, how are you?",
        converted_value_data_type="text",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Use await to handle the asynchronous call
    response = await hf_chat.send_prompt_async(prompt_request=prompt_request)  # type: ignore

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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(hf_chat.send_prompt_async(prompt_request=prompt_request))
        loop.close()

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
