# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Backend models package.

Pydantic models for API requests and responses.
"""

from pyrit.backend.models.attacks import (
    AddMessageRequest,
    AddMessageResponse,
    AttackListResponse,
    AttackMessagesResponse,
    AttackSummary,
    CreateAttackRequest,
    CreateAttackResponse,
    Message,
    MessagePiece,
    MessagePieceRequest,
    PrependedMessageRequest,
    Score,
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
    ConverterPreviewRequest,
    ConverterPreviewResponse,
    CreateConverterRequest,
    CreateConverterResponse,
    PreviewStep,
)
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)

__all__ = [
    # Attacks
    "AddMessageRequest",
    "AddMessageResponse",
    "AttackListResponse",
    "AttackMessagesResponse",
    "AttackSummary",
    "CreateAttackRequest",
    "CreateAttackResponse",
    "Message",
    "MessagePiece",
    "MessagePieceRequest",
    "PrependedMessageRequest",
    "Score",
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
    "ConverterPreviewRequest",
    "ConverterPreviewResponse",
    "CreateConverterRequest",
    "CreateConverterResponse",
    "PreviewStep",
    # Targets
    "CreateTargetRequest",
    "CreateTargetResponse",
    "TargetInstance",
    "TargetListResponse",
]
