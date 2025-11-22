# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
from typing import List, Optional

from pyrit.common import apply_defaults
from pyrit.common.path import DATASETS_PATH, SCORER_CONFIG_PATH
from pyrit.executor.attack.core.attack_config import (
    AttackAdversarialConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.core.attack_strategy import AttackStrategy
from pyrit.executor.attack.single_turn.role_play import RolePlayAttack, RolePlayPaths
from pyrit.models import SeedDataset
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget
from pyrit.scenarios.atomic_attack import AtomicAttack
from pyrit.scenarios.scenario import Scenario
from pyrit.scenarios.scenario_strategy import (
    ScenarioCompositeStrategy,
    ScenarioStrategy,
)
from pyrit.score.true_false.self_ask_true_false_scorer import (
    SelfAskTrueFalseScorer,
)


class ScamStrategy(ScenarioStrategy):
    """
    Strategies for the Scam Scenario.
    """

    ALL = ("all", {"all"})

    # Types of scam strategies
    ROLE_PLAY = ("role_play", {"role_play"})


class ScamScenario(Scenario):
    """
    ScamScenario is a preconfigured scenario which currently evaluates a model's
    ability to generate persuasive scam scripts for various scam types.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The ScamStrategy enum class.
        """
        return ScamStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: ScamStrategy.ALL (all scam strategies).
        """
        return ScamStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        objectives: Optional[List[str]] = None,
        objective_scorer: Optional[SelfAskTrueFalseScorer] = None,
        adversarial_chat: Optional[PromptChatTarget] = None,
        include_baseline: bool = True,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the ScamScenario.

        Args:
            objectives (Optional[List[str]]): List of objectives to test for scam-related harms.
            objective_scorer (Optional[SelfAskTrueFalseScorer]): Custom scorer for objective
                evaluation.
            adversarial_chat (Optional[PromptChatTarget]): Chat target used to rephrase the
                objective into the role-play context (in single-turn strategies).
            include_baseline (bool): Whether to include a baseline atomic attack that sends all objectives
                without modifications. Defaults to True. When True, a "baseline" attack is automatically
                added as the first atomic attack, allowing comparison between unmodified prompts and
                encoding-modified prompts.
            scenario_result_id (Optional[str]): Optional ID of an existing scenario result to resume.
        """
        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)
        self._adversarial_config = AttackAdversarialConfig(target=adversarial_chat)

        self._adversarial_chat = adversarial_chat or self._get_default_adversarial_target()
        self._objectives = objectives if objectives else self._get_default_dataset()

        super().__init__(
            name="Scam Scenario",
            version=self.version,
            strategy_class=ScamStrategy,
            objective_scorer_identifier=objective_scorer.get_identifier(),
            include_default_baseline=include_baseline,
            scenario_result_id=scenario_result_id,
        )

    def _get_default_objective_scorer(self) -> SelfAskTrueFalseScorer:
        """
        Build the default conversational evaluation scorer.

        Returns:
            SelfAskTrueFalseScorer: Scorer that checks if the assistant response contains scam script.
        """
        scam_script_evaluator = SelfAskTrueFalseScorer(
            chat_target=OpenAIChatTarget(),
            true_false_question_path=SCORER_CONFIG_PATH / "true_false_question" / "scam_scripts.yaml",
        )

        return scam_script_evaluator

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        """
        Provide an OpenAI target for the role-play rephrasing step.

        Returns:
            OpenAIChatTarget: Target that supplies the persuasion script rephrasing.
        """
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            temperature=1.2,
        )

    def _get_default_dataset(self) -> List[str]:
        """
        Load the default scam seed prompts.

        Returns:
            List[str]: Prompts from the scam seed dataset.
        """
        seed_prompts: List[str] = []
        scam_path = pathlib.Path(DATASETS_PATH) / "seed_prompts" / "scams.prompt"
        seed_prompts.extend(SeedDataset.from_yaml_file(scam_path).get_values())
        return seed_prompts

    async def _get_atomic_attack_from_strategy_async(self, strategy: str) -> AtomicAttack:
        """
        Translate the strategy into an actual AtomicAttack.

        Args:
            strategy: The CyberStrategy enum (SingleTurn or MultiTurn).

        Returns:
            AtomicAttack configured for the specified strategy.
        """
        # objective_target is guaranteed to be non-None by parent class validation
        assert self._objective_target is not None
        attack_strategy: Optional[AttackStrategy] = None

        if strategy == "role_play":
            attack_strategy = RolePlayAttack(
                objective_target=self._objective_target,
                adversarial_chat=self._adversarial_chat,
                role_play_definition_path=RolePlayPaths.PERSUASION_SCRIPT.value,
                attack_scoring_config=self._scorer_config,
            )
        else:
            raise ValueError(f"Unknown ScamStrategy: {strategy}")

        return AtomicAttack(
            atomic_attack_name=f"scam_{strategy}",
            attack=attack_strategy,
            objectives=self._objectives,
            memory_labels=self._memory_labels,
        )

    async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
        """
        Generate atomic attacks for each strategy.

        Returns:
            List[AtomicAttack]: List of atomic attacks to execute.
        """
        atomic_attacks: List[AtomicAttack] = []
        strategies = ScenarioCompositeStrategy.extract_single_strategy_values(
            composites=self._scenario_composites, strategy_type=ScamStrategy
        )

        for strategy in strategies:
            atomic_attacks.append(await self._get_atomic_attack_from_strategy_async(strategy))

        return atomic_attacks
