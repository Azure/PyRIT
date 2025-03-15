# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains common utilities for PyRIT."""

from pyrit.common.default_values import get_non_required_value, get_required_value
from pyrit.common.display_response import display_image_response
from pyrit.common.download_hf_model import (
    download_chunk,
    download_file,
    download_files,
    download_specific_files,
    get_available_files,
)
from pyrit.common.initialization import (
    initialize_pyrit,
    AZURE_SQL,
    DUCK_DB,
    IN_MEMORY,
)
from pyrit.common.net_utility import get_httpx_client, make_request_and_raise_if_error_async
from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.common.print import print_chat_messages_with_color
from pyrit.common.singleton import Singleton
from pyrit.common.utils import combine_dict, combine_list
from pyrit.common.yaml_loadable import YamlLoadable

__all__ = [
    "AZURE_SQL",
    "DUCK_DB",
    "IN_MEMORY",
    "combine_dict",
    "combine_list",
    "display_image_response",
    "download_chunk",
    "download_file",
    "download_files",
    "download_specific_files",
    "get_available_files",
    "get_httpx_client",
    "get_non_required_value",
    "get_required_value",
    "initialize_pyrit",
    "is_in_ipython_session",
    "make_request_and_raise_if_error_async",
    "print_chat_messages_with_color",
    "Singleton",
    "YamlLoadable",
]
