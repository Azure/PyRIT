# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import AzureOpenAI

from pyrit.embedding._text_embedding import _TextEmbedding


class AzureTextEmbedding(_TextEmbedding):
    def __init__(self, *, api_key: str, api_base: str, api_version: str = "2023-05-15", model: str) -> None:
        """Generate embedding using the Azure API

        Args:
            api_key: The API key to use
            api_base: The API base to use
            model: The engine to use, usually name of the deployment
            api_version: The API version to use
        """
        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base,
            azure_deployment=model,
        )
        self._model = model
        super().__init__()
