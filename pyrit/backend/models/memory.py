# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Memory query response models.

Models for messages, scores, attack results, scenario results, and seeds.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.models import PromptDataType, PromptResponseError

# ============================================================================
# Message Queries
# ============================================================================


class MessageQueryResponse(BaseModel):
    """Response model for message piece queries."""

    id: str = Field(..., description="Message piece ID")
    conversation_id: str = Field(..., description="Parent conversation ID")
    sequence: int = Field(..., description="Sequence in conversation")
    role: str = Field(..., description="Message role")
    original_value: str = Field(..., description="Original content")
    original_value_data_type: PromptDataType = Field(..., description="Original data type")
    converted_value: str = Field(..., description="Converted content")
    converted_value_data_type: PromptDataType = Field(..., description="Converted data type")
    converter_identifiers: List[Dict[str, Any]] = Field(default_factory=list, description="Applied converters")
    target_identifier: Dict[str, Any] = Field(..., description="Target identifier (filtered)")
    labels: Optional[Dict[str, str]] = Field(None, description="Message labels")
    response_error: Optional[PromptResponseError] = Field(None, description="Error type if any")
    timestamp: datetime = Field(..., description="Message timestamp")


# ============================================================================
# Score Queries
# ============================================================================

ScoreType = Literal["true_false", "float_scale", "unknown"]


class ScoreQueryResponse(BaseModel):
    """Response model for score queries."""

    id: str = Field(..., description="Score ID")
    message_piece_id: str = Field(..., description="Associated message piece ID")
    score_value: str = Field(..., description="Score value ('true'/'false' or numeric)")
    score_value_description: str = Field(..., description="Human-readable score description")
    score_type: ScoreType = Field(..., description="Type of score")
    score_category: Optional[List[str]] = Field(None, description="Score categories")
    score_rationale: str = Field(..., description="Explanation for the score")
    scorer_identifier: Dict[str, Any] = Field(..., description="Scorer identifier (filtered)")
    objective: Optional[str] = Field(None, description="Scoring objective")
    timestamp: datetime = Field(..., description="Score timestamp")


# ============================================================================
# Attack Results
# ============================================================================

AttackOutcome = Literal["success", "failure", "undetermined"]


class AttackResultQueryResponse(BaseModel):
    """Response model for attack result queries."""

    id: str = Field(..., description="Attack result ID")
    conversation_id: str = Field(..., description="Associated conversation ID")
    objective: str = Field(..., description="Attack objective")
    attack_identifier: Dict[str, Any] = Field(..., description="Attack identifier (filtered)")
    outcome: Optional[str] = Field(None, description="Attack outcome (success, failure, undetermined)")
    outcome_reason: Optional[str] = Field(None, description="Explanation for outcome")
    executed_turns: int = Field(..., description="Number of turns executed")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: Optional[datetime] = Field(None, description="Result timestamp")


# ============================================================================
# Scenario Results
# ============================================================================

ScenarioRunState = Literal["CREATED", "IN_PROGRESS", "COMPLETED", "FAILED"]


class ScenarioResultQueryResponse(BaseModel):
    """Response model for scenario result queries."""

    id: str = Field(..., description="Scenario result ID")
    scenario_name: str = Field(..., description="Scenario name")
    scenario_description: Optional[str] = Field(None, description="Scenario description")
    scenario_version: int = Field(..., description="Scenario version")
    pyrit_version: str = Field(..., description="PyRIT version used")
    run_state: ScenarioRunState = Field(..., description="Current run state")
    objective_target_identifier: Dict[str, Any] = Field(..., description="Target identifier (filtered)")
    labels: Optional[Dict[str, str]] = Field(None, description="Scenario labels")
    number_tries: int = Field(..., description="Number of objectives attempted")
    completion_time: Optional[datetime] = Field(None, description="Completion timestamp")
    timestamp: datetime = Field(..., description="Creation timestamp")


# ============================================================================
# Seeds
# ============================================================================

SeedType = Literal["prompt", "objective", "simulated_conversation"]


class SeedQueryResponse(BaseModel):
    """Response model for seed queries."""

    id: str = Field(..., description="Seed ID")
    value: str = Field(..., description="Seed content")
    data_type: PromptDataType = Field(..., description="Content data type")
    name: Optional[str] = Field(None, description="Seed name")
    dataset_name: Optional[str] = Field(None, description="Dataset name")
    seed_type: SeedType = Field(..., description="Type of seed")
    harm_categories: Optional[List[str]] = Field(None, description="Harm categories")
    description: Optional[str] = Field(None, description="Seed description")
    source: Optional[str] = Field(None, description="Seed source")
    date_added: Optional[datetime] = Field(None, description="Date added")
