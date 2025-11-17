# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, List, Optional, Sequence, Type, TypeVar

from pyrit.common import apply_defaults
from pyrit.executor.attack import (
    AttackScoringConfig,
    AttackStrategy,
    PromptSendingAttack,
)
from pyrit.executor.attack.multi_turn.multi_prompt_sending import (
    MultiPromptSendingAttack,
)
from pyrit.executor.attack.single_turn.many_shot_jailbreak import (
    ManyShotJailbreakAttack,
)
from pyrit.executor.attack.single_turn.role_play import RolePlayAttack, RolePlayPaths
from pyrit.memory.central_memory import CentralMemory
from pyrit.models.seed_group import SeedGroup
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget
from pyrit.scenarios import (
    AtomicAttack,
    Scenario,
    ScenarioStrategy,
)
from pyrit.score import (
    SelfAskRefusalScorer,
    TrueFalseInverterScorer,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

AttackStrategyT = TypeVar("AttackStrategyT", bound=AttackStrategy)


class ContentHarmStrategy(ScenarioStrategy):
    """
    ContentHarmStrategy defines a set of strategies for testing model behavior
    in several different harm categories. The scenario is designed to provide quick
    feedback on model performance with respect to common harm types with the idea being that
    users will dive deeper into specific harm categories based on initial results.

    Each tag represents a different harm category that the model can be tested for.
    Specifying the all tag will include a comprehensive test suite covering all harm categories.
    Users should define objective datasets in CentralMemory corresponding to each harm category
    they wish to test which can then be reused across multiple runs of the scenario.
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


class ContentHarmScenario(Scenario):
    """

    Content Harm Scenario implementation for PyRIT.

    This scenario contains various harm-based checks that you can run to get a quick idea about model behavior
    with respect to certain harm categories.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> Type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The ContentHarmStrategy enum class.
        """
        return ContentHarmStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: ContentHarmStrategy.ALL
        """
        return ContentHarmStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        objective_scorer: Optional[TrueFalseScorer] = None,
        adversarial_chat: Optional[PromptChatTarget] = None,
        seed_dataset_prefix: Optional[str] = None,
        scenario_result_id: Optional[str] = None,
    ):
        """
        Initialize the Content Harm Scenario.

        Args:
            adversarial_chat (Optional[PromptChatTarget]): Additionally used for scoring defaults.
                If not provided, a default OpenAI target will be created using environment variables.
            objective_scorer (Optional[TrueFalseScorer]): Scorer to evaluate attack success.
                If not provided, creates a default composite scorer using Azure Content Filter
                and SelfAsk Refusal scorers.
                seed_dataset_prefix (Optional[str]): Prefix of the dataset to use to retrieve the objectives.
                This will be used to retrieve the appropriate seed groups from CentralMemory. If not provided,
                defaults to "content_harm".
            max_concurrency (int): Maximum number of concurrent operations. Defaults to 10.
            max_retries (int): Maximum number of automatic retries if the scenario raises an exception.
                Set to 0 (default) for no automatic retries. If set to a positive number,
                the scenario will automatically retry up to this many times after an exception.
                For example, max_retries=3 allows up to 4 total attempts (1 initial + 3 retries).
        """

        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)
        self._adversarial_chat = adversarial_chat if adversarial_chat else self._get_default_adversarial_target()
        self._objective_scorer = objective_scorer if objective_scorer else self._get_default_scorer()
        self._seed_dataset_prefix = seed_dataset_prefix

        super().__init__(
            name="Content Harm Scenario",
            version=self.version,
            objective_scorer_identifier=self._objective_scorer.get_identifier(),
            strategy_class=ContentHarmStrategy,
            default_aggregate=ContentHarmStrategy.ALL,
            scenario_result_id=scenario_result_id,
        )

    def _get_strategy_seeds_groups(self, seed_dataset_prefix: Optional[str] = None) -> Dict[str, Sequence[SeedGroup]]:
        """
        Get the objectives from the provided seed dataset name from central memory

        If a seed dataset prefix is provided, it is used directly with the harm strategy name
         appended to the end to retrieve the objectives for each harm strategy.
         For example, if the seed_dataset_prefix is "scenario_harm" and the harm strategy is
         "hate", the dataset name used to retrieve objectives will be "scenario_harm_hate". If no
         seed dataset name is provided, the default "content_harm" is used.

        Args:
            seed_dataset_prefix (Optional[str]): The provided seed dataset name.
        Returns:
            Dict[str, List[str]]: A dictionary which maps harms to the seed groups retrieved from
            the seed dataset in CentralMemory.
        Raises:
            ValueError: If no objectives are found in the specified dataset or the dataset cannot
            be found.
        """
        memory = CentralMemory.get_memory_instance()
        if not seed_dataset_prefix:
            seed_dataset_prefix = "content_harm"
        seeds_by_strategy = {}
        selected_harms = {comp.strategies[0].value for comp in self._scenario_composites if comp.strategies}
        for harm_strategy in selected_harms:
            harm_dataset_name = seed_dataset_prefix + "_" + harm_strategy
            strategy_seed_groups = memory.get_seed_groups(dataset_name=harm_dataset_name)
            strategy_objectives: list[str] = [
                obj.objective.value for obj in strategy_seed_groups if obj.objective is not None
            ]
            if len(strategy_objectives) == 0:
                raise ValueError(
                    f"No objectives found for {harm_strategy} in the dataset {harm_dataset_name}.\n"
                    f"Ensure that the dataset is properly loaded into CentralMemory and follows the naming "
                    f"schema seed_dataset_prefix + _ + {harm_strategy}."
                )
            seeds_by_strategy[harm_strategy] = strategy_seed_groups
        return seeds_by_strategy

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            temperature=1.0,
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
        selected_harms = {comp.strategies[0].value for comp in self._scenario_composites if comp.strategies}
        seeds = self._get_strategy_seeds_groups()
        for strategy in selected_harms:
            atomic_attacks.extend(self._get_strategy_attacks(strategy=strategy, seed_groups=seeds[strategy]))
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
            objective = seed_group.objective.value if seed_group.objective is not None else None
            if objective:
                strategy_seed_objectives.append(objective)
                # strategy_prompt_sequence.append(objective)

            # create new SeedGroup without the objective for PromptSendingAttack
            strategy_seed_group_prompt_only.append(SeedGroup(prompts=seed_group.prompts))
            for prompt in seed_group.prompts:
                strategy_prompt_sequence.append(prompt.value)

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
