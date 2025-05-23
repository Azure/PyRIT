# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score

ResultT = TypeVar("ResultT", bound="AttackResult")


@dataclass
class AttackResult:
    """Base class for all attack results"""

    conversation_id: str
    objective: str
    orchestrator_identifier: dict[str, str]
    last_response: Optional[PromptRequestPiece] = None
    last_score: Optional[Score] = None
    achieved_objective: bool = False
    executed_turns: int = 0
