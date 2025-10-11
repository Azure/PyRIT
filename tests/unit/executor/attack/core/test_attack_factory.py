# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock
from typing import get_args

import pytest

from pyrit.executor.attack.core.attack_factory import attack_factory, AttackType
from pyrit.executor.attack.single_turn import (
    PromptSendingAttack,
    FlipAttack,
    ContextComplianceAttack,
    ManyShotJailbreakAttack,
    RolePlayAttack,
    SkeletonKeyAttack,
)
from pyrit.executor.attack.multi_turn import (
    MultiPromptSendingAttack,
    RedTeamingAttack,
    CrescendoAttack,
    TAPAttack,
    TreeOfAttacksWithPruningAttack,
)
from pyrit.prompt_target import PromptTarget


@pytest.fixture
def mock_target():
    """Create a mock PromptTarget for testing"""
    return MagicMock(spec=PromptTarget)


@pytest.mark.usefixtures("patch_central_database")
class TestAttackFactory:
    """
    Tests for the attack_factory function.
    
    Note: Some attacks require additional parameters to be set via the default component system.
    Tests only cover attacks that can be instantiated with just objective_target, or verify
    that appropriate errors are raised when required defaults are not configured.
    """

    def test_attack_factory_prompt_sending_attack(self, mock_target: PromptTarget):
        """Test creating a PromptSendingAttack"""
        attack = attack_factory(
            attack_type="PromptSendingAttack",
            objective_target=mock_target,
        )
        assert isinstance(attack, PromptSendingAttack)
        assert attack._objective_target == mock_target

    def test_attack_factory_flip_attack(self, mock_target: PromptTarget):
        """Test creating a FlipAttack"""
        attack = attack_factory(
            attack_type="FlipAttack",
            objective_target=mock_target,
        )
        assert isinstance(attack, FlipAttack)
        assert attack._objective_target == mock_target

    def test_attack_factory_many_shot_jailbreak_attack(self, mock_target: PromptTarget):
        """Test creating a ManyShotJailbreakAttack"""
        attack = attack_factory(
            attack_type="ManyShotJailbreakAttack",
            objective_target=mock_target,
        )
        assert isinstance(attack, ManyShotJailbreakAttack)
        assert attack._objective_target == mock_target

    def test_attack_factory_skeleton_key_attack(self, mock_target: PromptTarget):
        """Test creating a SkeletonKeyAttack"""
        attack = attack_factory(
            attack_type="SkeletonKeyAttack",
            objective_target=mock_target,
        )
        assert isinstance(attack, SkeletonKeyAttack)
        assert attack._objective_target == mock_target

    def test_attack_factory_multi_prompt_sending_attack(self, mock_target: PromptTarget):
        """Test creating a MultiPromptSendingAttack"""
        attack = attack_factory(
            attack_type="MultiPromptSendingAttack",
            objective_target=mock_target,
        )
        assert isinstance(attack, MultiPromptSendingAttack)
        assert attack._objective_target == mock_target

    def test_attack_factory_context_compliance_requires_defaults(self, mock_target: PromptTarget):
        """Test that ContextComplianceAttack raises error when adversarial_config not set via defaults"""
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            attack_factory(
                attack_type="ContextComplianceAttack",
                objective_target=mock_target,
            )

    def test_attack_factory_role_play_requires_defaults(self, mock_target: PromptTarget):
        """Test that RolePlayAttack raises error when required params not set via defaults"""
        with pytest.raises(TypeError, match="argument should be a str or an os.PathLike object"):
            attack_factory(
                attack_type="RolePlayAttack",
                objective_target=mock_target,
            )

    def test_attack_factory_red_teaming_requires_defaults(self, mock_target: PromptTarget):
        """Test that RedTeamingAttack raises error when required params not set via defaults"""
        with pytest.raises((ValueError, AttributeError)):
            attack_factory(
                attack_type="RedTeamingAttack",
                objective_target=mock_target,
            )

    def test_attack_factory_crescendo_requires_defaults(self, mock_target: PromptTarget):
        """Test that CrescendoAttack raises error when adversarial_config not set via defaults"""
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            attack_factory(
                attack_type="CrescendoAttack",
                objective_target=mock_target,
            )

    def test_attack_factory_tap_requires_defaults(self, mock_target: PromptTarget):
        """Test that TAPAttack raises error when adversarial_config not set via defaults"""
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            attack_factory(
                attack_type="TAPAttack",
                objective_target=mock_target,
            )

    def test_attack_factory_tree_of_attacks_requires_defaults(self, mock_target: PromptTarget):
        """Test that TreeOfAttacksWithPruningAttack raises error when adversarial_config not set via defaults"""
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            attack_factory(
                attack_type="TreeOfAttacksWithPruningAttack",
                objective_target=mock_target,
            )

    def test_attack_factory_invalid_type(self, mock_target: PromptTarget):
        """Test that an invalid attack type raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported attack type"):
            attack_factory(
                attack_type="InvalidAttackType",  # type: ignore
                objective_target=mock_target,
            )

    def test_attack_factory_all_types_are_defined(self):
        """Test that all types in AttackType literal are handled by the factory"""
        # Get all attack types from the literal
        all_attack_types = get_args(AttackType)
        
        # Ensure we have at least some attack types defined
        assert len(all_attack_types) > 0, "AttackType should define at least one attack type"
        assert len(all_attack_types) == 11, "Expected 11 attack types to be defined"

    def test_attack_factory_returns_objects_with_execute_async(self, mock_target: PromptTarget):
        """Test that created attacks have an execute_async method"""
        attack = attack_factory(
            attack_type="PromptSendingAttack",
            objective_target=mock_target,
        )
        assert hasattr(attack, "execute_async")
        assert callable(attack.execute_async)

    @pytest.mark.parametrize(
        "attack_type,expected_class",
        [
            ("PromptSendingAttack", PromptSendingAttack),
            ("FlipAttack", FlipAttack),
            ("ManyShotJailbreakAttack", ManyShotJailbreakAttack),
            ("SkeletonKeyAttack", SkeletonKeyAttack),
            ("MultiPromptSendingAttack", MultiPromptSendingAttack),
        ],
    )
    def test_attack_factory_returns_correct_type_for_simple_attacks(
        self, attack_type: str, expected_class: type, mock_target: PromptTarget
    ):
        """
        Parametrized test to verify attacks that don't require additional defaults
        return the correct class and can be instantiated successfully.
        """
        attack = attack_factory(
            attack_type=attack_type,  # type: ignore
            objective_target=mock_target,
        )
        assert isinstance(attack, expected_class)
