# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    ConversationSession,
    MultiTurnAttackContext,
)
from pyrit.models import ConversationType


def _make_context() -> MultiTurnAttackContext:
    return MultiTurnAttackContext(
        params=AttackParameters(objective="Test objective"),
        session=ConversationSession(),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestRotateConversationForSingleTurnTarget:
    """Test the _rotate_conversation_for_single_turn_target helper."""

    def _make_strategy(self, *, supports_multi_turn: bool):
        """Create a minimal MultiTurnAttackStrategy for testing."""
        from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import MultiTurnAttackStrategy

        target = MagicMock()
        target.supports_multi_turn = supports_multi_turn
        target.get_identifier.return_value = MagicMock()

        with patch.multiple(
            MultiTurnAttackStrategy,
            __abstractmethods__=frozenset(),
            _perform_async=AsyncMock(),
            _setup_async=AsyncMock(),
        ):
            strategy = MultiTurnAttackStrategy(
                objective_target=target,
                context_type=MultiTurnAttackContext,
            )
        return strategy  # noqa: RET504

    def test_noop_for_multi_turn_target(self):
        strategy = self._make_strategy(supports_multi_turn=True)
        context = _make_context()
        context.executed_turns = 1
        original_id = context.session.conversation_id

        strategy._rotate_conversation_for_single_turn_target(context=context)

        assert context.session.conversation_id == original_id
        assert len(context.related_conversations) == 0

    def test_noop_on_first_turn(self):
        strategy = self._make_strategy(supports_multi_turn=False)
        context = _make_context()
        context.executed_turns = 0
        original_id = context.session.conversation_id

        strategy._rotate_conversation_for_single_turn_target(context=context)

        assert context.session.conversation_id == original_id
        assert len(context.related_conversations) == 0

    def test_rotates_on_second_turn_for_single_turn_target(self):
        strategy = self._make_strategy(supports_multi_turn=False)
        context = _make_context()
        context.executed_turns = 1
        original_id = context.session.conversation_id

        strategy._rotate_conversation_for_single_turn_target(context=context)

        assert context.session.conversation_id != original_id
        assert len(context.related_conversations) == 1
        ref = next(iter(context.related_conversations))
        assert ref.conversation_id == original_id
        assert ref.conversation_type == ConversationType.PRUNED

    def test_rotates_multiple_turns(self):
        strategy = self._make_strategy(supports_multi_turn=False)
        context = _make_context()
        seen_ids = {context.session.conversation_id}

        for turn in range(1, 4):
            context.executed_turns = turn
            strategy._rotate_conversation_for_single_turn_target(context=context)
            assert context.session.conversation_id not in seen_ids
            seen_ids.add(context.session.conversation_id)

        assert len(context.related_conversations) == 3


@pytest.mark.usefixtures("patch_central_database")
class TestValueErrorGuards:
    """Test that incompatible attacks raise ValueError for single-turn targets."""

    def _make_single_turn_target(self):
        target = MagicMock()
        target.supports_multi_turn = False
        target.get_identifier.return_value = MagicMock()
        return target

    def _make_adversarial_config(self):
        from pyrit.executor.attack.core.attack_config import AttackAdversarialConfig

        adversarial_chat = MagicMock()
        adversarial_chat.get_identifier.return_value = MagicMock()
        return AttackAdversarialConfig(target=adversarial_chat)

    def _make_scoring_config(self):
        from pyrit.executor.attack.core.attack_config import AttackScoringConfig
        from pyrit.score import TrueFalseScorer

        scorer = MagicMock(spec=TrueFalseScorer)
        scorer.get_identifier.return_value = MagicMock()
        return AttackScoringConfig(objective_scorer=scorer)

    @pytest.mark.asyncio
    async def test_crescendo_raises_for_single_turn_target(self):
        from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack, CrescendoAttackContext

        target = self._make_single_turn_target()
        attack = CrescendoAttack(
            objective_target=target,
            attack_adversarial_config=self._make_adversarial_config(),
            attack_scoring_config=self._make_scoring_config(),
        )

        context = CrescendoAttackContext(
            params=AttackParameters(objective="Test"),
        )

        with pytest.raises(ValueError, match="CrescendoAttack requires a multi-turn target"):
            await attack._setup_async(context=context)

    @pytest.mark.asyncio
    async def test_multi_prompt_sending_raises_for_single_turn_target(self):
        from pyrit.executor.attack.multi_turn.multi_prompt_sending import MultiPromptSendingAttack

        target = self._make_single_turn_target()
        attack = MultiPromptSendingAttack(objective_target=target)

        context = MultiTurnAttackContext(
            params=AttackParameters(objective="Test"),
        )

        with pytest.raises(ValueError, match="MultiPromptSendingAttack requires a multi-turn target"):
            await attack._setup_async(context=context)
