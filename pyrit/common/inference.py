# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

from pyrit.models import (
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsageInformation,
    PromptResponse,
)


def text_to_prompt_response(
    text: str,
    model_name: str,
    completion_token_count: int = 0,
    prompt_token_count: int = 0,
    total_token_count: int = 0,
) -> PromptResponse:
    """
    Convert a text response to a proper PromptResponse object.

    This is a wrapper around the OpenAI text completion object so that our code conforms to their API response.
    See https://platform.openai.com/docs/guides/completion/introduction for more info

    Args:
        text: The text contained in the response
        model_name: The model used to generate the response
        completion_token_count: The number of tokens used in the completion
        prompt_token_count: The number of tokens sent in the prompt
        total_token_count: Total number of tokens used in the request

    Returns:
        Fully formed PromptResponse object.
    """
    current_epoch_time = int(time.time())
    prompt_response = PromptResponse(
        completion=text,
        model=model_name,
        object="text_completion",
        completion_tokens=completion_token_count,
        prompt_tokens=prompt_token_count,
        total_tokens=total_token_count,
        created_at=current_epoch_time,
        finish_reason="stop",
        api_request_time_to_complete_ns=0,
    )
    return prompt_response


def embedding_to_embedding_response(
    embedding: list[float],
    model_name: str,
    prompt_token_count: int = 0,
    total_token_count: int = 0,
) -> EmbeddingResponse:
    """
    Convert a raw embedding to a proper EmbeddingResponse object.

    This is a wrapper around the OpenAI embedding response object so that our code conforms
    to their API response. See https://platform.openai.com/docs/guides/embeddings/what-are-embeddings

    Args:
        embedding: The raw embedding
        model_name: The model used to create the embedding
        prompt_token_count: The number of tokens in the prompt
        total_token_count: The total number of tokens in the prompt and the response

    Returns:
        Fully formed EmbeddingResponse object.
    """
    usage_information = EmbeddingUsageInformation(prompt_tokens=prompt_token_count, total_tokens=total_token_count)
    embedding_data = EmbeddingData(embedding=embedding, index=0, object="embedding")
    response = EmbeddingResponse(model=model_name, object="list", usage=usage_information, data=[embedding_data])
    return response
