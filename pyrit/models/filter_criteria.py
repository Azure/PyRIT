# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

PromptConverterState = Literal["original", "converted"]


@dataclass
class PromptFilterCriteria:
    """
    An object that contains properties that can be used to filter memory queries to PromptReequestPieces.
    """

    orchestrator_id: Optional[str | uuid.UUID] = None
    conversation_id: Optional[str | uuid.UUID] = None
    prompt_ids: Optional[list[str] | list[uuid.UUID]] = None
    labels: Optional[dict[str, str]] = None
    sent_after: Optional[datetime] = None
    sent_before: Optional[datetime] = None
    original_values: Optional[list[str]] = None
    converted_values: Optional[list[str]] = None
    data_type: Optional[str] = None
    not_data_type: Optional[str] = None
    converted_value_sha256: Optional[list[str]] = None
