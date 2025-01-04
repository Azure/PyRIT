# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal

PromptDataType = Literal["text", "image_path", "audio_path", "video_path", "url", "error"]

"""
The type of the error in the prompt response
blocked: blocked by an external filter e.g. Azure Filters
none: no exception is raised
processing: there is an exception thrown unrelated to the query
unknown: the type of error is unknown
"""
PromptResponseError = Literal["blocked", "none", "processing", "empty", "unknown"]

ChatMessageRole = Literal["system", "user", "assistant"]
