# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Any, Dict


class OpenAIToolType(str, Enum):
    WEB_SEARCH_PREVIEW = "web_search_preview"
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"


def web_search_tool() -> Dict[str, Any]:
    return {"type": OpenAIToolType.WEB_SEARCH_PREVIEW.value}


def code_interpreter_tool() -> Dict[str, Any]:
    return {"type": OpenAIToolType.CODE_INTERPRETER.value}


def file_search_tool() -> Dict[str, Any]:
    return {"type": OpenAIToolType.FILE_SEARCH.value}
