# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models import (
    ChatMessageListDictContent,
    PromptRequestPiece,
    PromptRequestResponse,
)
from pyrit.prompt_target.aws_bedrock_claude_chat_target import (
    AWSBedrockClaudeChatTarget,
)


def is_boto3_installed():
    try:
        import boto3  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.fixture
def aws_target() -> AWSBedrockClaudeChatTarget:
    return AWSBedrockClaudeChatTarget(
        model_id="anthropic.claude-v2",
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        enable_ssl_verification=True,
    )


@pytest.fixture
def mock_prompt_request():
    request_piece = PromptRequestPiece(
        role="user", original_value="Hello, Claude!", converted_value="Hello, how are you?"
    )
    return PromptRequestResponse(request_pieces=[request_piece])


@pytest.mark.skipif(not is_boto3_installed(), reason="boto3 is not installed")
@pytest.mark.asyncio
async def test_send_prompt_async(aws_target, mock_prompt_request):
    with patch("boto3.client", new_callable=MagicMock) as mock_boto:
        mock_client = mock_boto.return_value
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=MagicMock(return_value=json.dumps({"content": [{"text": "I'm good, thanks!"}]})))
        }

        response = await aws_target.send_prompt_async(prompt_request=mock_prompt_request)

        assert response.request_pieces[0].converted_value == "I'm good, thanks!"


@pytest.mark.skipif(not is_boto3_installed(), reason="boto3 is not installed")
@pytest.mark.asyncio
async def test_validate_request_valid(aws_target, mock_prompt_request):
    aws_target._validate_request(prompt_request=mock_prompt_request)


@pytest.mark.skipif(not is_boto3_installed(), reason="boto3 is not installed")
@pytest.mark.asyncio
async def test_validate_request_invalid_data_type(aws_target):
    request_pieces = [
        PromptRequestPiece(
            role="user", original_value="test", converted_value="ImageData", converted_value_data_type="video_path"
        )
    ]
    invalid_request = PromptRequestResponse(request_pieces=request_pieces)

    with pytest.raises(ValueError, match="This target only supports text and image_path."):
        aws_target._validate_request(prompt_request=invalid_request)


@pytest.mark.skipif(not is_boto3_installed(), reason="boto3 is not installed")
@pytest.mark.asyncio
async def test_complete_chat_async(aws_target):
    with patch("boto3.client", new_callable=MagicMock) as mock_boto:
        mock_client = mock_boto.return_value
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=MagicMock(return_value=json.dumps({"content": [{"text": "Test Response"}]})))
        }

        response = await aws_target._complete_chat_async(
            messages=[ChatMessageListDictContent(role="user", content=[{"type": "text", "text": "Test input"}])]
        )

        assert response == "Test Response"
