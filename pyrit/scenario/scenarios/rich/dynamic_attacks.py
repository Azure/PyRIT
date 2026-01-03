# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict, List, Optional

from pyrit.executor.attack import (
    AttackScoringConfig,
    AttackStrategy,
    PromptSendingAttack,
)
from pyrit.executor.attack.single_turn.flip_attack import FlipAttack
from pyrit.executor.attack.single_turn.role_play import RolePlayAttack, RolePlayPaths
from pyrit.models import SeedGroup, SeedObjective
from pyrit.prompt_target import PromptChatTarget
from pyrit.scenario.core.atomic_attack import AtomicAttack


@dataclass
class AtomicAttackWithMetrics:
    """
    Container for an AtomicAttack with its associated performance metrics.

    Attributes:
        atomic_attack: The atomic attack instance.
        success_probability: Estimated probability of attack success (0.0 to 1.0).
        avg_execution_time: Average execution time in seconds.
    """

    atomic_attack: AtomicAttack
    success_probability: float
    avg_execution_time: float
    


class DynamicAttacks:
    """
    Produces a list of AtomicAttack instances with associated success probability
    and average execution time metrics.

    This class is responsible for constructing atomic attacks and returning them
    with their estimated performance metrics.
    """

    def __init__(
        self,
        *,
        objectives: list[str],
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        scorer_config: Optional[AttackScoringConfig] = None,
        memory_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize DynamicAttacks with target and configuration.

        Args:
            objectives (list[str]): List of objective strings to create attacks for.
            objective_target (PromptChatTarget): The target to attack.
            adversarial_chat (PromptChatTarget): The adversarial chat target used for attacks
                that require rephrasing objectives (e.g., RolePlayAttack).
            scorer_config (Optional[AttackScoringConfig]): Configuration for scoring attacks.
            memory_labels (Optional[Dict[str, str]]): Labels to apply to prompts for tracking.
        """
        self._objective_target = objective_target
        self._adversarial_chat = adversarial_chat
        self._objectives = objectives
        self._scorer_config = scorer_config
        self._memory_labels = memory_labels or {}

        # Create seed groups from objectives
        self._seed_groups = self._create_seed_groups_from_objectives(objectives=objectives)

    def _create_seed_groups_from_objectives(
        self,
        *,
        objectives: List[str],
    ) -> List[SeedGroup]:
        """
        Create SeedGroup instances from a list of objective strings.

        Args:
            objectives (List[str]): List of objective strings.

        Returns:
            List[SeedGroup]: List of SeedGroup instances, one per objective.
        """
        seed_groups: List[SeedGroup] = []
        for objective in objectives:
            seed_objective = SeedObjective(value=objective)
            seed_group = SeedGroup(seeds=[seed_objective])
            seed_groups.append(seed_group)
        return seed_groups

    def get_attacks_with_metrics(
        self,
        *,
        seed_groups: List[SeedGroup],
    ) -> List[AtomicAttackWithMetrics]:
        """
        Retrieve the list of AtomicAttack instances with their metrics.

        Args:
            seed_groups (List[SeedGroup]): The seed groups to use for attacks.

        Returns:
            List[AtomicAttackWithMetrics]: List of atomic attacks with their metrics.
        """
        return self._get_strategy_attacks(seed_groups=seed_groups)
    
    def _create_attack_list(self) -> list[AtomicAttack]:
        """
        Creates a list of Atomic Attack instances without SeedPrompts.
        
        :param self: Description
        """
        self._attacks = list[AtomicAttack]()


        # Flip Attack
        flip_attack = FlipAttack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
        )
        self._attacks.append(
            AtomicAttack(
                atomic_attack_name="flip_attack",
                attack=flip_attack,
                seed_groups=self._seed_groups,
                memory_labels=self._memory_labels,
            )
        )

        # Role Play
        for role_play in [RolePlayPaths.MOVIE_SCRIPT, RolePlayPaths.TRIVIA_GAME, RolePlayPaths.VIDEO_GAME]:
            role_play_attack = RolePlayAttack(
                objective_target=self._objective_target,
                adversarial_chat=self._adversarial_chat,
                attack_scoring_config=self._scorer_config,
                role_play_definition_path=role_play.value,
            )
            self._attacks.append(
                AtomicAttack(
                    atomic_attack_name=f"role_play_{role_play.name.lower()}",
                    attack=role_play_attack,
                    seed_groups=self._seed_groups,
                    memory_labels=self._memory_labels,
                )
            )

        # Context Compliance


    def _get_strategy_attacks(
        self,
        *,
        seed_groups: List[SeedGroup],
    ) -> List[AtomicAttackWithMetrics]:
        """
        Create AtomicAttack instances with their associated metrics.

        Args:
            seed_groups (List[SeedGroup]): The seed groups to use for attacks.

        Returns:
            List[AtomicAttackWithMetrics]: List of atomic attacks with their metrics.
        """
        attacks_with_metrics: List[AtomicAttackWithMetrics] = []


        atomic_attack = AtomicAttack(
            atomic_attack_name="prompt_sending",
            attack=prompt_sending_attack,
            seed_groups=list(seed_groups),
            memory_labels=self._memory_labels,
        )

        # Standin values for success_probability and avg_execution_time
        # TODO: Replace with actual computed values
        success_probability = 0.5
        avg_execution_time = 1.0

        attacks_with_metrics.append(
            AtomicAttackWithMetrics(
                atomic_attack=atomic_attack,
                success_probability=success_probability,
                avg_execution_time=avg_execution_time,
            )
        )

        return attacks_with_metrics
