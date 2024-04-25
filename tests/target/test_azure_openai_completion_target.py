# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import AsyncMock, patch

from pyrit.models import PromptRequestPiece, PromptRequestResponse
import pytest
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

from pyrit.prompt_target import AzureOpenAICompletionTarget


@pytest.fixture
def openai_mock_return() -> Completion:
    return Completion(
        id="12345678-1a2b-3c4e5f-a123-12345678abcd",
        object="text_completion",
        choices=[
            CompletionChoice(
                index=0,
                text="hi",
                finish_reason="stop",
                logprobs=None,
            )
        ],
        created=1629389505,
        model="gpt-35-turbo",
        usage=CompletionUsage(
            prompt_tokens=1,
            total_tokens=2,
            completion_tokens=1,
        ),
    )


@pytest.fixture
def azure_completion_target() -> AzureOpenAICompletionTarget:
    return AzureOpenAICompletionTarget(
        deployment_name="gpt-35-turbo",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )


@pytest.fixture
def prompt_request_response() -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="1234",
                original_prompt_text="hello",
                converted_prompt_text="hello",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            )
        ]
    )


@pytest.mark.asyncio
async def test_azure_complete_async_return(
    openai_mock_return: Completion,
    azure_completion_target: AzureOpenAICompletionTarget,
    prompt_request_response: PromptRequestResponse,
):
    with patch("openai.resources.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = openai_mock_return
        prompt_request_response: PromptRequestResponse = await azure_completion_target.send_prompt_async(
            prompt_request=prompt_request_response
        )
        assert len(prompt_request_response.request_pieces) == 1
        assert prompt_request_response.request_pieces[0].converted_prompt_text == "hi"


def test_azure_invalid_key_raises():
    os.environ[AzureOpenAICompletionTarget.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAICompletionTarget(
            deployment_name="gpt-4",
            endpoint="https://mock.azure.com/",
            api_key="",
            api_version="some_version",
        )


def test_azure_initialization_with_no_deployment_raises():
    os.environ[AzureOpenAICompletionTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAICompletionTarget()


def test_azure_invalid_endpoint_raises():
    os.environ[AzureOpenAICompletionTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAICompletionTarget(
            deployment_name="gpt-4",
            endpoint="",
            api_key="xxxxx",
            api_version="some_version",
        )
