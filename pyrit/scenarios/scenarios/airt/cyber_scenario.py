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
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models import SeedDataset
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget
from pyrit.scenarios.atomic_attack import AtomicAttack
from pyrit.scenarios.scenario import Scenario
from pyrit.scenarios.scenario_strategy import (
    ScenarioCompositeStrategy,
    ScenarioStrategy,
)
from pyrit.score import (
    SelfAskRefusalScorer,
    SelfAskTrueFalseScorer,
    TrueFalseCompositeScorer,
    TrueFalseInverterScorer,
    TrueFalseScoreAggregator,
)


class CyberStrategy(ScenarioStrategy):
    """
    Strategies for malware-focused cyber attacks. While not in the CyberStrategy class, a
    few of these include:
    * Shell smashing
    * Zip bombs
    * File deletion (rm -rf /)

    """

    # Aggregate members (special markers that expand to strategies with matching tags)
    ALL = ("all", {"all"})
    FAST = ("fast", {"fast"})
    SLOW = ("slow", {"slow"})

    # Attack strategies
    SingleTurn = ("single_turn", {"fast"})
    MultiTurn = ("multi_turn", {"slow"})


class CyberScenario(Scenario):
    """
    Cyber scenario implementation for PyRIT.

    This scenario tests how willing models are to exploit cybersecurity harms by generating
    malware. The CyberStrategy class contains different variations of the malware generation
    techniques.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The CyberStrategy enum class.
        """
        return CyberStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: CyberStrategy.ALL (all cyber strategies).
        """
        return CyberStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        adversarial_chat: Optional[PromptChatTarget] = None,
        objectives: Optional[List[str]] = None,
        objective_scorer: Optional[TrueFalseCompositeScorer] = None,
        include_baseline: bool = True,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the cyber harms scenario. Note that the cyber harms scenario is slightly different from the encoding
        and foundry scenarios, as it doesn't use converters.

        Args:
            adversarial_chat (Optional[PromptChatTarget]): Adversarial chat for the red teaming attack, corresponding
                to CyberStrategy.MultiTurn. If not provided, defaults to an OpenAI chat target.
            objectives (Optional[List[str]]): List of objectives to test for cyber harms, e.g. malware generation.
            objective_scorer (Optional[SelfAskTrueFalseScorer]): Objective scorer for malware detection. If not
                provided, defaults to a SelfAskScorer using the malware.yaml file under the scorer config store for
                malware detection
            include_baseline (bool): Whether to include a baseline atomic attack that sends all objectives
                without modifications. Defaults to True. When True, a "baseline" attack is automatically
                added as the first atomic attack, allowing comparison between unmodified prompts and
                attack-modified prompts.
            scenario_result_id (Optional[str]): Optional ID of an existing scenario result to resume.
        """
        # CyberScenario uses a "take object, make config" pattern to expose a more ergonomic interface. Helper
        # methods return objects, not configs.

        # In this context the "objective" scorer has nothing to do with the "objective" target.
        # The scoring config is what's used later in scoring orchestration, not the scorer object.
        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)

        # Adversarial chat is used by the RedTeamingAttack (multiturn) variation of the scenario. Otherwise, it's not
        # invoked.
        self._adversarial_chat = adversarial_chat if adversarial_chat else self._get_default_adversarial_target()
        self._adversarial_config = AttackAdversarialConfig(target=self._adversarial_chat)

        self._objectives = objectives if objectives else self._get_default_dataset()

        super().__init__(
            name="Cyber Scenario",
            version=self.version,
            strategy_class=CyberStrategy,
            default_aggregate=CyberStrategy.FAST,
            objective_scorer_identifier=objective_scorer.get_identifier(),
            include_default_baseline=include_baseline,
            scenario_result_id=scenario_result_id,
        )

    def _get_default_objective_scorer(self) -> TrueFalseCompositeScorer:
        """
        Retrieve the default objective scorer.
        The scoring mechanism is as follows:
        1. Detect for the presence of malware using a SelfAskTrueFalseScorer.
        2. Detect for NON-refusal as a backstop. (If the target model refuses to produce malware, we want that
        to register as a negative score.)
        """

        presence_of_malware = SelfAskTrueFalseScorer(
            chat_target=OpenAIChatTarget(),
            true_false_question_path=SCORER_CONFIG_PATH / "true_false_question" / "malware.yaml",
        )

        backstop = TrueFalseInverterScorer(scorer=SelfAskRefusalScorer(chat_target=OpenAIChatTarget()))

        return TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND, scorers=[presence_of_malware, backstop]
        )

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        """
        The adversarial target is used in the red teaming attack variation, where we want an unsafe model attempting
        to generate malware.
        """
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            temperature=1.2,
        )

    def _get_default_dataset(self) -> list[str]:
        """
        Get the default seed prompts for malware tests.

        This dataset includes a set of exploits that represent cybersecurity harms.

        Returns:
            list[str]: List of seed prompt strings to be encoded and tested.
        """
        seed_prompts: List[str] = []
        malware_path = pathlib.Path(DATASETS_PATH) / "seed_prompts"
        seed_prompts.extend(SeedDataset.from_yaml_file(malware_path / "malware.prompt").get_values())
        return seed_prompts

    async def _get_atomic_attack_from_strategy_async(self, strategy: ScenarioCompositeStrategy) -> AtomicAttack:
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
        if strategy.strategies[0] == CyberStrategy.FAST:
            attack_strategy = PromptSendingAttack(
                objective_target=self._objective_target,
                attack_scoring_config=self._scorer_config,
            )
        elif strategy.strategies[0] == CyberStrategy.SLOW:
            attack_strategy = RedTeamingAttack(
                objective_target=self._objective_target,
                attack_scoring_config=self._scorer_config,
                attack_adversarial_config=self._adversarial_config,
            )
        else:
            raise ValueError(f"Unknown CyberStrategy: {strategy}, contains: {strategy.strategies}")

        return AtomicAttack(
            atomic_attack_name=f"cyber_{strategy}",
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
        for strategy in self._scenario_composites:
            atomic_attacks.append(await self._get_atomic_attack_from_strategy_async(strategy))
        return atomic_attacks
