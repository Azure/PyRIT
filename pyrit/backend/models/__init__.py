# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backend models package.

Pydantic models for API requests and responses.
"""

from pyrit.backend.models.common import (
    ALLOWED_IDENTIFIER_FIELDS,
    SENSITIVE_FIELD_PATTERNS,
    FieldError,
    IdentifierDict,
    PaginatedResponse,
    PaginationInfo,
    ProblemDetail,
    filter_sensitive_fields,
)
from pyrit.backend.models.conversations import (
    BranchConversationRequest,
    BranchConversationResponse,
    ConversationResponse,
    ConverterConfig,
    ConvertersResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    MessagePieceInput,
    MessagePieceResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    SetConvertersRequest,
    SetSystemPromptRequest,
    SystemPromptResponse,
)
from pyrit.backend.models.converters import (
    ConversionStep,
    ConverterListResponse,
    ConverterMetadataResponse,
    PreviewConverterRequest,
    PreviewConverterResponse,
)
from pyrit.backend.models.memory import (
    AttackResultQueryResponse,
    MessageQueryResponse,
    ScenarioResultQueryResponse,
    ScoreQueryResponse,
    SeedQueryResponse,
)
from pyrit.backend.models.registry import (
    InitializerListResponse,
    InitializerMetadataResponse,
    ScenarioListResponse,
    ScenarioMetadataResponse,
    ScorerListResponse,
    ScorerMetadataResponse,
    TargetListResponse,
    TargetMetadataResponse,
)

__all__ = [
    # Common
    "ALLOWED_IDENTIFIER_FIELDS",
    "SENSITIVE_FIELD_PATTERNS",
    "FieldError",
    "filter_sensitive_fields",
    "IdentifierDict",
    "PaginatedResponse",
    "PaginationInfo",
    "ProblemDetail",
    # Conversations
    "BranchConversationRequest",
    "BranchConversationResponse",
    "ConversationResponse",
    "ConverterConfig",
    "ConvertersResponse",
    "CreateConversationRequest",
    "CreateConversationResponse",
    "MessagePieceInput",
    "MessagePieceResponse",
    "MessageResponse",
    "SendMessageRequest",
    "SendMessageResponse",
    "SetConvertersRequest",
    "SetSystemPromptRequest",
    "SystemPromptResponse",
    # Converters
    "ConversionStep",
    "ConverterListResponse",
    "ConverterMetadataResponse",
    "PreviewConverterRequest",
    "PreviewConverterResponse",
    # Memory
    "AttackResultQueryResponse",
    "MessageQueryResponse",
    "ScenarioResultQueryResponse",
    "ScoreQueryResponse",
    "SeedQueryResponse",
    # Registry
    "InitializerListResponse",
    "InitializerMetadataResponse",
    "ScenarioListResponse",
    "ScenarioMetadataResponse",
    "ScorerListResponse",
    "ScorerMetadataResponse",
    "TargetListResponse",
    "TargetMetadataResponse",
]
