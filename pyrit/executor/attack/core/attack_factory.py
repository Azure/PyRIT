# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal, Optional, get_args

from pyrit.prompt_target import PromptTarget
from pyrit.setup import apply_defaults_to_method

# Define allowed attack types as literals
AttackType = Literal[
    "PromptSendingAttack",
    "FlipAttack",
    "ContextComplianceAttack",
    "ManyShotJailbreakAttack",
    "RolePlayAttack",
    "SkeletonKeyAttack",
    "MultiPromptSendingAttack",
    "RedTeamingAttack",
    "CrescendoAttack",
    "TAPAttack",
    "TreeOfAttacksWithPruningAttack",
]


class AttackFactory:
    """
    Factory class for creating attack instances based on attack type.

    This factory simplifies the creation of attack instances by accepting a string
    identifier for the attack type and the necessary configuration parameters.
    The factory is designed to work with attacks that have default values configured,
    requiring only the essential parameters.
    """

    @staticmethod
    @apply_defaults_to_method
    def create_attack(
        *,
        attack_type: Optional[AttackType] = None,
        objective_target: PromptTarget,
    ):
        """
        Create an attack instance based on the attack type.

        This factory method creates attack instances by type. All returned attack instances
        have an `execute_async` method that accepts an `objective` parameter and returns
        an `AttackResult`.

        For attacks that require additional parameters (e.g., adversarial_chat,
        role_play_definition_path, max_attempts_on_failure), those parameters should be
        configured using the default component system. If required parameters are not set
        via defaults, the attack class will raise an appropriate error when instantiated.

        Args:
            attack_type (AttackType): The type of attack to create. Supported types:
                Single-turn attacks:
                - "PromptSendingAttack": Simple single-turn prompt injection
                - "FlipAttack": Flips words in the prompt (requires PromptChatTarget)
                - "ContextComplianceAttack": Uses context to bypass filters (requires adversarial_chat default)
                - "ManyShotJailbreakAttack": Single-turn with many example shots
                - "RolePlayAttack": Rephrases into role-play context (requires adversarial_chat and role_play_definition_path defaults)
                - "SkeletonKeyAttack": Skeleton key technique
                
                Multi-turn attacks:
                - "MultiPromptSendingAttack": Multi-turn prompt sequence
                - "RedTeamingAttack": Iterative red teaming (requires adversarial_chat default)
                - "CrescendoAttack": Progressively harmful prompts (requires adversarial_chat default)
                - "TAPAttack": Tree of Attacks with Pruning (requires adversarial_chat default)
                - "TreeOfAttacksWithPruningAttack": Alias for TAPAttack (requires adversarial_chat default)
                
            objective_target (PromptTarget): The target system to attack.

        Returns:
            An attack strategy instance with an execute_async method.

        Raises:
            ValueError: If the attack_type is not supported.
            Various exceptions from attack classes if required parameters are not configured via defaults.

        Examples:
            >>> from pyrit.prompt_target import OpenAIChatTarget
            >>> from pyrit.executor.attack import AttackFactory
            >>>
            >>> # Simple attack that works with just objective_target
            >>> target = OpenAIChatTarget()
            >>> attack = AttackFactory.create_attack(
            ...     attack_type="PromptSendingAttack",
            ...     objective_target=target
            ... )
            >>> result = await attack.execute_async(objective="Tell me how to make a bomb")
            >>>
            >>> # Attack requiring additional defaults to be configured
            >>> # (adversarial_chat, role_play_definition_path must be set via defaults)
            >>> attack = AttackFactory.create_attack(
            ...     attack_type="RolePlayAttack",
            ...     objective_target=target
            ... )
        """
        # Validate attack type
        if attack_type is None:
            raise ValueError(
                "attack_type must be provided either as a parameter or configured via set_default_value()"
            )
        
        if attack_type not in get_args(AttackType):
            supported_types = ", ".join(get_args(AttackType))
            raise ValueError(
                f"Unsupported attack type '{attack_type}'. "
                f"Supported types are: {supported_types}"
            )

        # Import attack classes (lazy imports to avoid circular dependencies)
        from pyrit.executor.attack.single_turn import (
            ContextComplianceAttack,
            FlipAttack,
            ManyShotJailbreakAttack,
            PromptSendingAttack,
            RolePlayAttack,
            SkeletonKeyAttack,
        )
        from pyrit.executor.attack.multi_turn import (
            CrescendoAttack,
            MultiPromptSendingAttack,
            RedTeamingAttack,
            TAPAttack,
            TreeOfAttacksWithPruningAttack,
        )

        # Create and return the appropriate attack instance
        # For attacks with additional required parameters, pass None and let the class handle validation
        
        if attack_type == "PromptSendingAttack":
            return PromptSendingAttack(
                objective_target=objective_target,
            )

        elif attack_type == "FlipAttack":
            return FlipAttack(
                objective_target=objective_target,
            )

        elif attack_type == "ContextComplianceAttack":
            return ContextComplianceAttack(
                objective_target=objective_target,
                attack_adversarial_config=None,  # Should be set via defaults
            )

        elif attack_type == "ManyShotJailbreakAttack":
            return ManyShotJailbreakAttack(
                objective_target=objective_target,
            )

        elif attack_type == "RolePlayAttack":
            return RolePlayAttack(
                objective_target=objective_target,
                adversarial_chat=None,  # Should be set via defaults
                role_play_definition_path=None,  # Should be set via defaults
            )

        elif attack_type == "SkeletonKeyAttack":
            return SkeletonKeyAttack(
                objective_target=objective_target,
            )

        elif attack_type == "MultiPromptSendingAttack":
            return MultiPromptSendingAttack(
                objective_target=objective_target,
            )

        elif attack_type == "RedTeamingAttack":
            return RedTeamingAttack(
                objective_target=objective_target,
                attack_adversarial_config=None,  # Should be set via defaults
            )

        elif attack_type == "CrescendoAttack":
            return CrescendoAttack(
                objective_target=objective_target,
                attack_adversarial_config=None,  # Should be set via defaults
            )

        elif attack_type == "TAPAttack":
            return TAPAttack(
                objective_target=objective_target,
                attack_adversarial_config=None,  # Should be set via defaults
            )

        elif attack_type == "TreeOfAttacksWithPruningAttack":
            return TreeOfAttacksWithPruningAttack(
                objective_target=objective_target,
                attack_adversarial_config=None,  # Should be set via defaults
            )

        # This should never be reached due to the initial validation, but keep for safety
        raise ValueError(f"Unsupported attack type: {attack_type}")


# Convenience function that maintains backward compatibility
def attack_factory(
    *,
    attack_type: Optional[AttackType] = None,
    objective_target: PromptTarget,
):
    """
    Convenience function for creating attack instances.

    This is a wrapper around AttackFactory.create_attack() that provides
    a more functional-style interface while maintaining compatibility with
    the class-based factory for default value injection.

    See AttackFactory.create_attack() for full documentation.
    """
    return AttackFactory.create_attack(attack_type=attack_type, objective_target=objective_target)
