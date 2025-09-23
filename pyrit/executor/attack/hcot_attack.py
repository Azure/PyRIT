# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackStrategy,
    SingleTurnAttackContext,
)
from pyrit.models import AttackResult, AttackOutcome
from pyrit.memory import CentralMemory


class HCoTAttack(SingleTurnAttackStrategy):
    """
    Hijacking Chain-of-Thought (H-CoT) jailbreak attack.
    Simulates adversarial prompts that hijack a model's reasoning process
    to reveal otherwise restricted information.
    """

    def __init__(self, *, max_turns: int = 1):
        # Pass context_type so the parent class initializes correctly
        super().__init__(context_type=SingleTurnAttackContext)
        self.max_turns = max_turns
        self._memory = CentralMemory.get_memory_instance()

    # --- Required abstract methods ---
    async def _setup_async(self, context: SingleTurnAttackContext):
        """Setup any preconditions for the attack."""
        return context

    async def _perform_async(self, context: SingleTurnAttackContext) -> AttackResult:
        """
        Main attack logic.
        For now, this is a placeholder returning UNDETERMINED outcome.
        """
        return AttackResult(
            outcome=AttackOutcome.UNDETERMINED,
            outcome_reason="HCoTAttack placeholder result",
            conversation_id=context.conversation_id,  # comes from the context
            objective=context.objective,              # ✅ required
            attack_identifier=self.__class__.__name__ # ✅ required
        )

    async def _teardown_async(self, context: SingleTurnAttackContext):
        """Clean up resources if needed."""
        return context

    def _validate_context(self, context: SingleTurnAttackContext) -> bool:
        """Validate the attack context (always true for now)."""
        return True
