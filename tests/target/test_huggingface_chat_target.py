# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from asyncio import Task
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from pyrit.prompt_target import HuggingFaceChatTarget
from pyrit.models.prompt_request_response import PromptRequestResponse, PromptRequestPiece


def is_torch_installed():
    try:
        import torch  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


# Fixture to mock get_required_value
@pytest.fixture(autouse=True)
def mock_get_required_value(request):
    if request.node.name != "test_init_with_no_token_var_raises":
        with patch(
            "pyrit.prompt_target.hugging_face.hugging_face_chat_target.default_values.get_required_value",
            return_value="dummy_token",
        ):
            yield
    else:
        # Do not apply the mock for this test
        yield


# Fixture to mock download_specific_files globally for all tests
@pytest.fixture(autouse=True)
def mock_download_specific_files():
    with patch("pyrit.common.download_hf_model.download_specific_files", return_value=None) as mock_download:
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


class AwaitableMock(AsyncMock):
    def __await__(self):
        return iter([])


@pytest.fixture(autouse=True)
def mock_create_task():
    with patch("asyncio.create_task", return_value=AwaitableMock(spec=Task)):
        yield


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_init_with_no_token_var_raises(monkeypatch):
    # Ensure the environment variable is unset
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    with pytest.raises(ValueError) as excinfo:
        HuggingFaceChatTarget(model_id="test_model", use_cuda=False, hf_access_token=None)

    assert "Environment variable HUGGINGFACE_TOKEN is required" in str(excinfo.value)


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
@pytest.mark.asyncio
async def test_initialization():
    # Test the initialization without loading the actual models
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    assert hf_chat.model_id == "test_model"
    assert not hf_chat.use_cuda
    assert hf_chat.device == "cpu"

    await hf_chat.load_model_and_tokenizer()
    assert hf_chat.model is not None
    assert hf_chat.tokenizer is not None


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_is_model_id_valid_true():
    # Simulate valid model ID
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    assert hf_chat.is_model_id_valid()


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_is_model_id_valid_false():
    # Simulate invalid model ID by causing an exception
    with patch("transformers.PretrainedConfig.from_pretrained", side_effect=Exception("Invalid model")):
        hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
        assert not hf_chat.is_model_id_valid()


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
@pytest.mark.asyncio
async def test_load_model_and_tokenizer():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    await hf_chat.load_model_and_tokenizer()
    assert hf_chat.model is not None
    assert hf_chat.tokenizer is not None


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
@pytest.mark.asyncio
async def test_send_prompt_async():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    await hf_chat.load_model_and_tokenizer()

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


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
@pytest.mark.asyncio
async def test_missing_chat_template_error():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False)
    await hf_chat.load_model_and_tokenizer()
    hf_chat.tokenizer.chat_template = None

    request_piece = PromptRequestPiece(
        role="user",
        original_value="Hello, how are you?",
        converted_value="Hello, how are you?",
        converted_value_data_type="text",
    )
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError) as excinfo:
        # Use await to handle the asynchronous call
        await hf_chat.send_prompt_async(prompt_request=prompt_request)

    assert "Tokenizer does not have a chat template" in str(excinfo.value)


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
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


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
@pytest.mark.asyncio
async def test_load_with_missing_files():
    hf_chat = HuggingFaceChatTarget(model_id="test_model", use_cuda=False, necessary_files=["file1", "file2"])
    await hf_chat.load_model_and_tokenizer()

    assert hf_chat.model is not None
    assert hf_chat.tokenizer is not None


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
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
