# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures a ROT13 converter.

This configuration file defines a single converter that can be used
as an additional converter in attack configurations.
"""

from pyrit.prompt_converter import ROT13Converter

# Define the converter to be loaded as an additional converter
additional_converter = ROT13Converter()
