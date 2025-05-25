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

    # Unique identifier of the conversation that produced this result
    conversation_id: str

    # Natural-language description of the attacker’s objective
    objective: str

    # Identifiers of the orchestrator that executed the attack (e.g., name, version)
    orchestrator_identifier: dict[str, str]

    # Model response generated in the final turn of the attack
    last_response: Optional[PromptRequestPiece] = None

    # Score assigned to the final response by a scorer component
    last_score: Optional[Score] = None

    # Indicates whether the attack achieved its objective
    achieved_objective: bool = False

    # Total number of turns that were executed
    executed_turns: int = 0
