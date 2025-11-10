# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List, Optional, Sequence, Type, TypeVar

from pyrit.common.apply_defaults import apply_defaults
from pyrit.executor.attack import (
    AttackStrategy,
    PromptSendingAttack,
)
from pyrit.executor.attack.core.attack_config import (
    AttackScoringConfig,
)
from pyrit.executor.attack.multi_turn.multi_prompt_sending import MultiPromptSendingAttack
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import MultiTurnAttackStrategy
from pyrit.memory.central_memory import CentralMemory
from pyrit.models.seed_group import SeedGroup
from pyrit.models.seed_objective import SeedObjective
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget
from pyrit.scenarios import (
    AtomicAttack,
    Scenario,
    ScenarioCompositeStrategy,
    ScenarioStrategy,
)
from pyrit.score import (
    SelfAskRefusalScorer,
    TrueFalseInverterScorer,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

AttackStrategyT = TypeVar("AttackStrategyT", bound=AttackStrategy)


class RapidResponseHarmStrategy(ScenarioStrategy):
    """
    RapidResponseHarmStrategy defines a set of strategies for testing model behavior
    in several different harm categories. The scenario is designed to provide quick
    feedback on model performance with respect to common harm types with the idea being that
    users will dive deeper into specific harm categories based on initial results.

    Each tag represents a different harm category that the model can be tested for.
    Specifying the all tag will include a comprehensive test suite covering all harm categories.
    Users should define objective datasets in CentralMemory corresponding to each harm category
    they wish to test which can then be reused across multiple runs of the scenario. For each
    harm category, the scenario will run both a PromptSendingAttack and MultiPromptSendingAttack
    to evaluate model behavior. Specific harm categories will run additional attack types that 
    are relevant to that strategy.
    TODO: Expand the specific attacks run for each harm strategy as needed.
    """

    ALL = ("all", {"all"})

    Hate = ("hate", set[str]())
    Fairness = ("fairness", set[str]())
    Violence = ("violence", set[str]())
    Sexual = ("sexual", set[str]())
    Harassment = ("harassment", set[str]())
    Misinformation = ("misinformation", set[str]())
    Leakage = ("leakage", set[str]())


class RapidResponseHarmScenario(Scenario):
    """

    Rapid Response Harm Scenario implementation for PyRIT.

    This scenario contains various harm-based checks that you can run to get a quick idea about model behavior
    with respect to certain harm categories.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> Type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The RapidResponseHarmStrategy enum class.
        """
        return RapidResponseHarmStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: RapidResponseHarmStrategy.ALL (easy difficulty strategies).
        """
        return RapidResponseHarmStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        scenario_strategies: Sequence[RapidResponseHarmStrategy | ScenarioCompositeStrategy] | None = None,
        objective_target: PromptChatTarget,
        objective_scorer: Optional[TrueFalseScorer] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        seed_dataset_name: Optional[str] = None,
        max_concurrency: int = 10,
        max_retries: int = 0,
    ):
        """
        Initialize the Rapid Response Harm Scenario.

        Args:
            scenario_strategies (Sequence[RapidResponseHarmStrategy | ScenarioCompositeStrategy] | None):
                The harm strategies or composite strategies to include in this scenario. If None,
                defaults to RapidResponseHarmStrategy.ALL.
            objective_scorer (Optional[TrueFalseScorer]): The scorer used to evaluate if the model
                successfully decoded the payload. Defaults to DecodingScorer with encoding_scenario
                category.
            objective_target (PromptChatTarget): The target model to test for harms vulnerabilities.
            memory_labels (Optional[Dict[str, str]]): Optional labels to attach to memory entries
                for tracking and filtering.
            seed_dataset_name (str): Name of the dataset to use to retrieve the objectives. This will be used
                to retrieve the appropriate seed groups from CentralMemory.
            max_concurrency (int): Maximum number of concurrent operations. Defaults to 10.
            max_retries (int): Maximum number of automatic retries if the scenario raises an exception.
                Set to 0 (default) for no automatic retries. If set to a positive number,
                the scenario will automatically retry up to this many times after an exception.
                For example, max_retries=3 allows up to 4 total attempts (1 initial + 3 retries).
        """

        self._objective_target = objective_target
        objective_scorer = objective_scorer or self._get_default_scorer()
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)
        self._memory_labels = memory_labels or {}

        self._rapid_response_harm_strategy_composition = RapidResponseHarmStrategy.prepare_scenario_strategies(
            scenario_strategies, default_aggregate=RapidResponseHarmStrategy.ALL
        )
        self._seeds = self._get_seeds(seed_dataset_name)

        super().__init__(
            name="Rapid Response Harm Scenario",
            version=self.version,
            memory_labels=memory_labels,
            max_concurrency=max_concurrency,
            objective_scorer_identifier=objective_scorer.get_identifier(),
            objective_target=objective_target,
            max_retries=max_retries,
        )

    def _get_seeds(self, seed_dataset_name: Optional[str] = None) -> Dict[str, List[SeedGroup]]:
        """
        Get the objectives from the provided seed dataset name from central memory

        If a seed dataset name is provided, it is used directly with the harm strategy name
         appended to the end to retrieve the objectives for each harm strategy.
         For example, if the seed_dataset_name is "scenario_harm" and the harm strategy is
         "hate", the dataset name used to retrieve objectives will be "scenario_harm_hate". If no
         seed dataset name is provided, the default "rapid_response_harm" is used.

        Args:
            seed_dataset_name (Optional[str]): The provided seed dataset name.
        Returns:
            Dict[str, List[str]]: The dictionary of objectives from the seed dataset which map to each harm.
        Raises:
            ValueError: If no objectives are found in the specified dataset or the dataset cannot
            be found.
        """
        memory = CentralMemory.get_memory_instance()
        if not seed_dataset_name:
            seed_dataset_name = "rapid_response_harm"
        seeds_by_strategy = {}
        for harm_strategy in self._rapid_response_harm_strategy_composition:
            harm_dataset_name = seed_dataset_name + "_" + harm_strategy.name
            strategy_seed_groups = memory.get_seed_groups(dataset_name=harm_dataset_name)
            strategy_objectives: list[str] = [obj.objective.value for obj in strategy_seed_groups if obj.objective is not None]
            if len(strategy_objectives) == 0:
                raise ValueError(
                    f"No objectives found for {harm_strategy.name} in the dataset {harm_dataset_name}.\n"
                    f"Ensure that the dataset is properly loaded into CentralMemory and follows the naming "
                    f"schema seed_dataset_name + _ + {harm_strategy.name}."
                )
            seeds_by_strategy[harm_strategy.name] = strategy_seed_groups
        return seeds_by_strategy

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            temperature=0.7,
        )

    def _get_default_scorer(self) -> TrueFalseInverterScorer:
        return TrueFalseInverterScorer(
            scorer=SelfAskRefusalScorer(
                chat_target=OpenAIChatTarget(
                    endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
                )
            ),
        )

    async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
        """
        Retrieve the list of AtomicAttack instances for harm strategies.

        Returns:
            List[AtomicAttack]: The list of AtomicAttack instances for harm strategies.
        """
        atomic_attacks: List[AtomicAttack] = []
        for strategy in self._rapid_response_harm_strategy_composition:
            atomic_attacks.extend(
                self._get_strategy_attacks(strategy=strategy, seed_groups=self._seeds[strategy.name])
            )
        return atomic_attacks

    def _get_strategy_attacks(
        self,
        strategy: ScenarioCompositeStrategy,
        seed_groups: List[SeedGroup],
    ) -> List[AtomicAttack]:
        """
        Create an AtomicAttack instances for a given harm strategy. PromptSendingAttack and 
        MuliturnAttack are run for all harm strategies. Select strategies may also use a specific
        attack type.

        Args:
            strategy (ScenarioStrategy): The strategy to create the attack from.
            seed_groups (List[SeedGroup]): The seed groups associated with the harm dataset.

        Returns:
            List[AtomicAttack]: The constructed AtomicAttack instances for each attack type.
        """
        prompt_sending_attack = PromptSendingAttack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
        )
        multi_turn_attack = MultiPromptSendingAttack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
        )
        
        strategy_objectives = []
        strategy_seed_prompts = []
        for seed_group in seed_groups:
            strategy_objectives.append(seed_group.objective.value if seed_group.objective is not None else None)
            strategy_seed_prompts.append(SeedGroup(prompts=seed_group.prompts))


        attacks = [AtomicAttack(
            atomic_attack_name=strategy.name,
            attack=prompt_sending_attack,
            objectives=strategy_objectives,
            memory_labels=self._memory_labels,
            seed_groups=strategy_seed_prompts
        ),
        AtomicAttack(
            atomic_attack_name=strategy.name,
            attack=multi_turn_attack,
            objectives=strategy_objectives,
            memory_labels=self._memory_labels,
            prompt_sequence=[prompts.prompts for prompts in strategy_seed_prompts],
        )]
        return attacks