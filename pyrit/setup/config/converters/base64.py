# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures a Base64 converter.

This configuration file defines a single converter that can be used
as an additional converter in attack configurations.
"""

from pyrit.prompt_converter import Base64Converter

# Define the converter to be loaded as an additional converter
additional_converter = Base64Converter()
