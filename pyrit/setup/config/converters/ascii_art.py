# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This configures an AsciiArt attack.

This configuration file defines the parameters for a PromptSendingAttack
that uses ASCII art conversion. It can be modified to adjust the converter
settings or add additional configurations.
"""

from pyrit.prompt_converter import AsciiArtConverter

additional_converter = AsciiArtConverter()