# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backend models package.

Pydantic models for API requests and responses.
"""

from pyrit.backend.models.attacks import (
    AttackDetail,
    AttackListResponse,
    AttackSummary,
    CreateAttackRequest,
    CreateAttackResponse,
    Message,
    MessagePiece,
    MessagePieceRequest,
    PrependedMessageRequest,
    Score,
    SendMessageRequest,
    SendMessageResponse,
    UpdateAttackRequest,
)
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
from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterInstanceListResponse,
    ConverterMetadataResponse,
    ConverterPreviewRequest,
    ConverterPreviewResponse,
    CreateConverterRequest,
    CreateConverterResponse,
    InlineConverterConfig,
    NestedConverterConfig,
    PreviewStep,
)
from pyrit.backend.models.registry import (
    InitializerListResponse,
    InitializerMetadataResponse,
    ScenarioListResponse,
    ScenarioMetadataResponse,
    ScorerListResponse,
    ScorerMetadataResponse,
    TargetMetadataResponse,
)
from pyrit.backend.models.registry import (
    TargetListResponse as RegistryTargetListResponse,
)
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)

__all__ = [
    # Attacks
    "AttackDetail",
    "AttackListResponse",
    "AttackSummary",
    "CreateAttackRequest",
    "CreateAttackResponse",
    "Message",
    "MessagePiece",
    "MessagePieceRequest",
    "PrependedMessageRequest",
    "Score",
    "SendMessageRequest",
    "SendMessageResponse",
    "UpdateAttackRequest",
    # Common
    "ALLOWED_IDENTIFIER_FIELDS",
    "SENSITIVE_FIELD_PATTERNS",
    "FieldError",
    "filter_sensitive_fields",
    "IdentifierDict",
    "PaginatedResponse",
    "PaginationInfo",
    "ProblemDetail",
    # Converters
    "ConverterInstance",
    "ConverterInstanceListResponse",
    "ConverterMetadataResponse",
    "ConverterPreviewRequest",
    "ConverterPreviewResponse",
    "CreateConverterRequest",
    "CreateConverterResponse",
    "InlineConverterConfig",
    "NestedConverterConfig",
    "PreviewStep",
    # Registry
    "InitializerListResponse",
    "InitializerMetadataResponse",
    "ScenarioListResponse",
    "ScenarioMetadataResponse",
    "ScorerListResponse",
    "ScorerMetadataResponse",
    "RegistryTargetListResponse",
    "TargetMetadataResponse",
    # Targets
    "CreateTargetRequest",
    "CreateTargetResponse",
    "TargetInstance",
    "TargetListResponse",
]
