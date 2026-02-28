# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, Optional


@dataclass(frozen=True)
class ConversationStats:
    """Lightweight aggregate statistics for a conversation.

    Used to build attack summaries without loading full message pieces.
    """

    PREVIEW_MAX_LEN: ClassVar[int] = 100

    message_count: int = 0
    last_message_preview: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[datetime] = None
