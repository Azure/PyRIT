# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.embedding._text_embedding import _TextEmbedding
from pyrit.models import EmbeddingResponse


class ClipEmbedding(_TextEmbedding):
    def __init__(self):
        pass

    def generate_text_embedding(self, text: str, **kwargs) -> EmbeddingResponse:
        """Generate text embedding

        Args:
            text: The text to generate the embedding for
            **kwargs: Additional arguments to pass to the function.

        Returns:
            The embedding response
        """
        raise NotImplementedError("Clip embedding is not implemented yet.")
