# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures an ANSI attack converter.

This configuration file defines a single converter that can be used
as an additional converter in attack configurations.
"""

from pyrit.prompt_converter import AnsiAttackConverter

additional_converter = AnsiAttackConverter()