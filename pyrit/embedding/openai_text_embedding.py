# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import OpenAI

from pyrit.embedding._text_embedding import _TextEmbedding


class OpenAiTextEmbedding(_TextEmbedding):
    def __init__(self, *, model: str, api_key: str) -> None:
        """Generate embedding using OpenAI API

        Args:
            api_version: The API version to use
            model: The model to use
            api_key: The API key to use
        """
        self._client = OpenAI(
            api_key=api_key,
        )
        self._model = model
        super().__init__()
