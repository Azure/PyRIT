# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backward compatibility wrapper for OpenAI Sora targets.

This module maintains backward compatibility by re-exporting OpenAISora1Target as OpenAISoraTarget.
For new code, use OpenAISora1Target or OpenAISora2Target directly.

- OpenAISora1Target: Legacy Azure/OpenAI Sora API (uses JSON body, /jobs endpoints)
- OpenAISora2Target: New OpenAI Sora API (uses multipart form-data, direct task endpoints)
"""

from pyrit.prompt_target.openai.openai_sora1_target import OpenAISora1Target

# Backward compatibility: OpenAISoraTarget defaults to Sora-1 (legacy API)
OpenAISoraTarget = OpenAISora1Target

__all__ = ["OpenAISoraTarget"]
