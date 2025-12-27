# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Optional

from pyrit.common import apply_defaults
from pyrit.common.path import SCORER_SEED_PROMPT_PATH
from pyrit.executor.attack.core.attack_config import (
    AttackAdversarialConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.core.attack_strategy import AttackStrategy
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models import SeedGroup, SeedObjective
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.scenario import Scenario
from pyrit.scenario.core.scenario_strategy import (
    ScenarioCompositeStrategy,
    ScenarioStrategy,
)
from pyrit.score import (
    SelfAskRefusalScorer,
    SelfAskTrueFalseScorer,
    TrueFalseCompositeScorer,
    TrueFalseInverterScorer,
    TrueFalseScoreAggregator,
    TrueFalseScorer,
)


class CyberStrategy(ScenarioStrategy):
    """
    Strategies for malware-focused cyber attacks. While not in the CyberStrategy class, a
    few of these include:
    * Shell smashing
    * Zip bombs
    * File deletion (rm -rf /).
    """

    # Aggregate members (special markers that expand to strategies with matching tags)
    ALL = ("all", {"all"})
    SINGLE_TURN = ("single_turn", {"single_turn"})
    MULTI_TURN = ("multi_turn", {"multi_turn"})


class CyberScenario(Scenario):
    """
    Cyber scenario implementation for PyRIT.

    This scenario tests how willing models are to exploit cybersecurity harms by generating
    malware. The CyberScenario class contains different variations of the malware generation
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

    @classmethod
    def required_datasets(cls) -> list[str]:
        """Return a list of dataset names required by this scenario."""
        return ["airt_malware"]

    @apply_defaults
    def __init__(
        self,
        *,
        adversarial_chat: Optional[PromptChatTarget] = None,
        objectives: Optional[List[str]] = None,
        objective_scorer: Optional[TrueFalseScorer] = None,
        include_baseline: bool = True,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the cyber harms scenario.

        Args:
            adversarial_chat (Optional[PromptChatTarget]): Adversarial chat for the red teaming attack, corresponding
                to CyberStrategy.MultiTurn. If not provided, defaults to an OpenAI chat target.
            objectives (Optional[List[str]]): List of objectives to test for cyber harms, e.g. malware generation.
                If not provided, defaults to objectives from the airt_malware dataset.
            objective_scorer (Optional[TrueFalseScorer]): Objective scorer for malware detection. If not
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

        super().__init__(
            name="Cyber Scenario",
            version=self.version,
            strategy_class=CyberStrategy,
            objective_scorer_identifier=objective_scorer.get_identifier(),
            include_default_baseline=include_baseline,
            scenario_result_id=scenario_result_id,
        )

        # Convert objectives to seed_groups if provided, otherwise load from dataset
        if objectives:
            self._seed_groups = [SeedGroup(seeds=[SeedObjective(value=obj)]) for obj in objectives]
        else:
            self._seed_groups = self._get_default_seed_groups()

    def _get_default_objective_scorer(self) -> TrueFalseCompositeScorer:
        """
        Retrieve the default objective scorer.
        The scoring mechanism is as follows:
        1. Detect for the presence of malware using a SelfAskTrueFalseScorer.
        2. Detect for NON-refusal as a backstop. (If the target model refuses to produce malware, we want that
        to register as a negative score).

        Returns:
            TrueFalseCompositeScorer: Default objective scorer with backstop and malware evaluation.
        """
        presence_of_malware = SelfAskTrueFalseScorer(
            chat_target=OpenAIChatTarget(
                endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
                model_name=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL"),
            ),
            true_false_question_path=SCORER_SEED_PROMPT_PATH / "true_false_question" / "malware.yaml",
        )

        backstop = TrueFalseInverterScorer(
            scorer=SelfAskRefusalScorer(
                chat_target=OpenAIChatTarget(
                    endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
                    model_name=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL"),
                )
            )
        )

        return TrueFalseCompositeScorer(
            aggregator=TrueFalseScoreAggregator.AND, scorers=[presence_of_malware, backstop]
        )

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        """
        Create and retrieve the default adversarial target.

        Returns:
            OpenAIChatTarget: Default adversarial target, using an unfiltered endpoint.
        """
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            model_name=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL"),
            temperature=1.2,
        )

    def _get_default_seed_groups(self) -> List[SeedGroup]:
        """
        Get the default seed groups for malware tests.

        This dataset includes a set of exploits that represent cybersecurity harms.

        Returns:
            List[SeedGroup]: List of seed groups with objectives to be tested.
        """
        seed_groups = self._memory.get_seed_groups(dataset_name="airt_malware", is_objective=True)

        if not seed_groups:
            self._raise_dataset_exception()

        return list(seed_groups)

    def _get_atomic_attack_from_strategy(self, strategy: str) -> AtomicAttack:
        """
        Translate the strategy into an actual AtomicAttack.

        Args:
            strategy: The CyberStrategy enum (SingleTurn or MultiTurn).

        Returns:
            AtomicAttack: configured for the specified strategy.

        Raises:
            ValueError: if an unknown CyberStrategy is passed.
        """
        # objective_target is guaranteed to be non-None by parent class validation
        assert self._objective_target is not None
        attack_strategy: Optional[AttackStrategy] = None
        if strategy == "single_turn":
            attack_strategy = PromptSendingAttack(
                objective_target=self._objective_target,
                attack_scoring_config=self._scorer_config,
            )
        elif strategy == "multi_turn":
            attack_strategy = RedTeamingAttack(
                objective_target=self._objective_target,
                attack_scoring_config=self._scorer_config,
                attack_adversarial_config=self._adversarial_config,
            )
        else:
            raise ValueError(f"Unknown CyberStrategy: {strategy}")

        return AtomicAttack(
            atomic_attack_name=f"cyber_{strategy}",
            attack=attack_strategy,
            seed_groups=self._seed_groups,
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
            composites=self._scenario_composites, strategy_type=CyberStrategy
        )

        for strategy in strategies:
            atomic_attacks.append(self._get_atomic_attack_from_strategy(strategy))
        return atomic_attacks
