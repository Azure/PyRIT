# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Attack executor module."""

from pyrit.executor.attack.core import (
    AttackStrategy,
    AttackContext,
    AttackConverterConfig,
    AttackParameters,
    AttackScoringConfig,
    AttackAdversarialConfig,
    AttackExecutor,
    AttackExecutorResult,
    PrependedConversationConfiguration,
)

from pyrit.executor.attack.single_turn import (
    SingleTurnAttackStrategy,
    SingleTurnAttackContext,
    PromptSendingAttack,
    FlipAttack,
    ContextComplianceAttack,
    ManyShotJailbreakAttack,
    RolePlayAttack,
    RolePlayPaths,
    SkeletonKeyAttack,
)

from pyrit.executor.attack.multi_turn import (
    ConversationSession,
    MultiTurnAttackStrategy,
    MultiTurnAttackContext,
    MultiPromptSendingAttack,
    MultiPromptSendingAttackParameters,
    RedTeamingAttack,
    RTASystemPromptPaths,
    CrescendoAttack,
    CrescendoAttackContext,
    CrescendoAttackResult,
    TAPAttack,
    TreeOfAttacksWithPruningAttack,
    TAPAttackContext,
    TAPAttackResult,
)

from pyrit.executor.attack.component import (
    ConversationManager,
    ConversationState,
    ObjectiveEvaluator,
    generate_simulated_conversation_async,
    SimulatedConversationResult,
    SimulatedTargetSystemPromptPaths,
)

# Import printer modules last to avoid circular dependencies
from pyrit.executor.attack.printer import ConsoleAttackResultPrinter, AttackResultPrinter, MarkdownAttackResultPrinter

__all__ = [
    "AttackStrategy",
    "AttackContext",
    "AttackParameters",
    "CrescendoAttack",
    "CrescendoAttackContext",
    "CrescendoAttackResult",
    "MultiPromptSendingAttack",
    "MultiPromptSendingAttackParameters",
    "TAPAttack",
    "TreeOfAttacksWithPruningAttack",
    "TAPAttackContext",
    "TAPAttackResult",
    "SingleTurnAttackStrategy",
    "SingleTurnAttackContext",
    "PromptSendingAttack",
    "FlipAttack",
    "ContextComplianceAttack",
    "ManyShotJailbreakAttack",
    "RolePlayAttack",
    "RolePlayPaths",
    "SkeletonKeyAttack",
    "ConversationSession",
    "MultiTurnAttackStrategy",
    "MultiTurnAttackContext",
    "RedTeamingAttack",
    "RTASystemPromptPaths",
    "ConsoleAttackResultPrinter",
    "MarkdownAttackResultPrinter",
    "AttackResultPrinter",
    "AttackConverterConfig",
    "AttackScoringConfig",
    "AttackAdversarialConfig",
    "ConversationManager",
    "ConversationState",
    "AttackExecutor",
    "ObjectiveEvaluator",
    "AttackExecutorResult",
    "PrependedConversationConfiguration",
    "generate_simulated_conversation_async",
    "SimulatedConversationResult",
    "SimulatedTargetSystemPromptPaths",
]
