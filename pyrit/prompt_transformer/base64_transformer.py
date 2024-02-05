# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64

from pyrit.prompt_transformer import PromptTransformer


class Base64Transformer(PromptTransformer):
    def transform(self, prompt: str) -> str:
        """
        Simple transformer that just base64 encodes the prompt
        """
        string_bytes = prompt.encode("utf-8")
        encoded_bytes = base64.b64encode(string_bytes)
        return encoded_bytes.decode("utf-8")
