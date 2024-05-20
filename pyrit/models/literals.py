# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal


PromptDataType = Literal["text", "image_path", "audio_path", "url", "error"]

"""
The type of the error in the prompt response
blocked: blocked by an external filter e.g. Azure Filters
model: the model refused to answer or request e.g. "I'm sorry..."
processing: there is an exception thrown unrelated to the query
unknown: the type of error is unknown
"""
PromptResponseError = Literal["none", "blocked", "error", "model", "processing", "unknown"]

ChatMessageRole = Literal["system", "user", "assistant"]
