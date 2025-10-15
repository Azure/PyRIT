# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures a Leetspeak converter.

This configuration file defines a single converter that can be used
as an additional converter in attack configurations.
"""

from pyrit.prompt_converter import LeetspeakConverter

# Define the converter to be loaded as an additional converter
additional_converter = LeetspeakConverter()
