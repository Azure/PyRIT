# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
from typing import Dict, List, Optional, Sequence, Type, TypeVar

from pyrit.common import apply_defaults
from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack import (
    AttackScoringConfig,
    AttackStrategy,
    ManyShotJailbreakAttack,
    MultiPromptSendingAttack,
    PromptSendingAttack,
    RolePlayAttack,
    RolePlayPaths,
)
from pyrit.models import SeedDataset, SeedGroup, SeedObjective, SeedPrompt
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.scenario import Scenario
from pyrit.scenario.core.scenario_strategy import (
    ScenarioCompositeStrategy,
    ScenarioStrategy,
)
from pyrit.score import SelfAskRefusalScorer, TrueFalseInverterScorer, TrueFalseScorer

AttackStrategyT = TypeVar("AttackStrategyT", bound=AttackStrategy)


class ContentHarmsStrategy(ScenarioStrategy):
    """
    ContentHarmsStrategy defines a set of strategies for testing model behavior
    across several different harm categories. The scenario is designed to provide quick
    feedback on model performance with respect to common harm types with the idea being that
    users will dive deeper into specific harm categories based on initial results.

    Each tag represents a different harm category that the model can be tested for.
    Specifying the all tag will include a comprehensive test suite covering all harm categories.
    Users can defined objectives for each harm category via seed datasets or use the default datasets
    provided with PyRIT.
    For each harm category, the scenario will run a RolePlayAttack, ManyShotJailbreakAttack,
    PromptSendingAttack, and RedTeamingAttack for each objective in the dataset.
    to evaluate model behavior.
    """

    ALL = ("all", {"all"})

    Hate = ("hate", set[str]())
    Fairness = ("fairness", set[str]())
    Violence = ("violence", set[str]())
    Sexual = ("sexual", set[str]())
    Harassment = ("harassment", set[str]())
    Misinformation = ("misinformation", set[str]())
    Leakage = ("leakage", set[str]())


class ContentHarmsScenario(Scenario):
    """

    Content Harms Scenario implementation for PyRIT.

    This scenario contains various harm-based checks that you can run to get a quick idea about model behavior
    with respect to certain harm categories.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> Type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The ContentHarmsStrategy enum class.
        """
        return ContentHarmsStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: ContentHarmsStrategy.ALL
        """
        return ContentHarmsStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        adversarial_chat: Optional[PromptChatTarget] = None,
        objective_scorer: Optional[TrueFalseScorer] = None,
        scenario_result_id: Optional[str] = None,
        objectives_by_harm: Optional[Dict[str, Sequence[SeedGroup]]] = None,
    ):
        """
        Initialize the Content Harms Scenario.

        Args:
            adversarial_chat (Optional[PromptChatTarget]): Additionally used for scoring defaults.
                If not provided, a default OpenAI target will be created using environment variables.
            objective_scorer (Optional[TrueFalseScorer]): Scorer to evaluate attack success.
                If not provided, creates a default composite scorer using Azure Content Filter
                and SelfAsk Refusal scorers.
                seed_dataset_prefix (Optional[str]): Prefix of the dataset to use to retrieve the objectives.
                This will be used to retrieve the appropriate seed groups from CentralMemory. If not provided,
                defaults to "content_harm".
            scenario_result_id (Optional[str]): Optional ID of an existing scenario result to resume.
            objectives_by_harm (Optional[Dict[str, Sequence[SeedGroup]]]): A dictionary mapping harm strategies
                to their corresponding SeedGroups. If not provided, default seed groups will be loaded from datasets.
        """
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)
        self._adversarial_chat = adversarial_chat if adversarial_chat else self._get_default_adversarial_target()

        self._objective_scorer = objective_scorer if objective_scorer else self._get_default_scorer()

        super().__init__(
            name="Content Harms Scenario",
            version=self.version,
            objective_scorer_identifier=self._objective_scorer.get_identifier(),
            strategy_class=ContentHarmsStrategy,
            scenario_result_id=scenario_result_id,
        )
        self._objectives_by_harm = objectives_by_harm

    def _get_objectives_by_harm(
        self, objectives_by_harm: Optional[Dict[str, Sequence[SeedGroup]]] = None
    ) -> Dict[str, Sequence[SeedGroup]]:
        """
        Retrieve SeedGroups for each harm strategy. If objectives_by_harm is provided for a given harm strategy,
        use that directly.
        Otherwise, load the default seed groups from datasets.

        Returns:
            Dict[str, Sequence[SeedGroup]]: A dictionary mapping harm strategies to their corresponding SeedGroups.
        """
        seeds_by_strategy = {}

        selected_harms = ScenarioCompositeStrategy.extract_single_strategy_values(
            self._scenario_composites, strategy_type=ContentHarmsStrategy
        )
        for harm_strategy in selected_harms:
            seeds_by_strategy[harm_strategy] = self._memory.get_seeds(
                is_objective=True,
                harm_categories=harm_strategy,
            )
                
        return seeds_by_strategy

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            temperature=1.2,
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
        selected_harms = ScenarioCompositeStrategy.extract_single_strategy_values(
            self._scenario_composites, strategy_type=ContentHarmsStrategy
        )
        merged_objectives_by_harm = self._get_objectives_by_harm(self._objectives_by_harm)
        for strategy in selected_harms:
            atomic_attacks.extend(
                self._get_strategy_attacks(strategy=strategy, seed_groups=merged_objectives_by_harm[strategy])
            )
        return atomic_attacks

    def _get_strategy_attacks(
        self,
        strategy: str,
        seed_groups: Sequence[SeedGroup],
    ) -> List[AtomicAttack]:
        """
        Create AtomicAttack instances for a given harm strategy. RolePlayAttack, ManyShotJailbreakAttack,
        PromptSendingAttack, and RedTeamingAttack are run for all harm strategies.

        Args:
            strategy (ScenarioCompositeStrategy): The strategy to create the attack from.
            seed_groups (List[SeedGroup]): The seed groups associated with the harm dataset.

        Returns:
            List[AtomicAttack]: The constructed AtomicAttack instances for each attack type.
        """
        # objective_target is guaranteed to be non-None by parent class validation
        assert self._objective_target is not None

        prompt_sending_attack = PromptSendingAttack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
        )

        role_play_attack = RolePlayAttack(
            objective_target=self._objective_target,
            adversarial_chat=self._adversarial_chat,
            role_play_definition_path=RolePlayPaths.MOVIE_SCRIPT.value,
        )

        many_shot_jailbreak_attack = ManyShotJailbreakAttack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
        )

        multi_prompt_sending_attack = MultiPromptSendingAttack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
        )

        # Extract seed objectives and seed prompts from seed groups
        strategy_seed_objectives = []
        strategy_seed_group_prompt_only = []
        # prompt sequence for multi prompt attack, includes objective followed by seed prompts
        strategy_prompt_sequence = []
        for seed_group in seed_groups:
            objectives = [seed.value for seed in seed_group.seeds if isinstance(seed, SeedObjective)]
            if objectives:
                strategy_seed_objectives.extend(objectives)

            # create new SeedGroup without the objective for PromptSendingAttack
            seed_prompts = []
            for prompt in seed_group.seeds:
                seed_prompts.append(SeedPrompt(prompt.value))
                strategy_prompt_sequence.append(prompt.value)
            strategy_seed_group_prompt_only.append(SeedGroup(seeds=seed_prompts))

        attacks = [
            AtomicAttack(
                atomic_attack_name=strategy,
                attack=prompt_sending_attack,
                objectives=strategy_seed_objectives,
                memory_labels=self._memory_labels,
                seed_groups=strategy_seed_group_prompt_only,
            ),
            AtomicAttack(
                atomic_attack_name=strategy,
                attack=role_play_attack,
                objectives=strategy_seed_objectives,
                memory_labels=self._memory_labels,
            ),
            AtomicAttack(
                atomic_attack_name=strategy,
                attack=many_shot_jailbreak_attack,
                objectives=strategy_seed_objectives,
                memory_labels=self._memory_labels,
            ),
            AtomicAttack(
                atomic_attack_name=strategy,
                attack=multi_prompt_sending_attack,
                objectives=strategy_seed_objectives,
                memory_labels=self._memory_labels,
                prompt_sequence=strategy_prompt_sequence,
            ),
        ]
        return attacks
