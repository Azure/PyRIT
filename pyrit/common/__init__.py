# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module contains common utilities for PyRIT."""

from pyrit.common.data_url_converter import convert_local_image_to_data_url
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
from pyrit.common.utils import combine_dict, combine_list, get_random_indices, warn_if_set, get_kwarg_param
from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.common.deprecation import deprecation_message

__all__ = [
    "AZURE_SQL",
    "DUCK_DB",
    "IN_MEMORY",
    "combine_dict",
    "combine_list",
    "convert_local_image_to_data_url",
    "display_image_response",
    "download_chunk",
    "download_file",
    "download_files",
    "download_specific_files",
    "get_available_files",
    "get_httpx_client",
    "get_non_required_value",
    "get_random_indices",
    "get_required_value",
    "initialize_pyrit",
    "is_in_ipython_session",
    "make_request_and_raise_if_error_async",
    "print_chat_messages_with_color",
    "Singleton",
    "YamlLoadable",
    "deprecation_message",
    "warn_if_set",
    "get_kwarg_param",
]
