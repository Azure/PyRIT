# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Embedding module for PyRI to provide Azure and OpenAI text embedding classes."""


from pyrit.embedding.azure_text_embedding import AzureTextEmbedding
from pyrit.embedding.openai_text_embedding import OpenAiTextEmbedding

__all__ = [
    "AzureTextEmbedding",
    "OpenAiTextEmbedding",
]
