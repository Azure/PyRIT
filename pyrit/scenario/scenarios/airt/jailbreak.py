# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
from typing import List, Optional

from pyrit.common import apply_defaults
from pyrit.datasets import TextJailBreak
from pyrit.executor.attack.core.attack_config import (
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.core.attack_strategy import AttackStrategy
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.executor.attack.single_turn.many_shot_jailbreak import ManyShotJailbreakAttack
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models import SeedAttackGroup
from pyrit.prompt_converter import TextJailbreakConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioCompositeStrategy, ScenarioStrategy
from pyrit.score import (
    SelfAskRefusalScorer,
    TrueFalseInverterScorer,
    TrueFalseScorer,
)


class JailbreakStrategy(ScenarioStrategy):
    """
    Strategy for jailbreak attacks.
    """

    # Aggregate members (special markers that expand to strategies with matching tags)
    ALL = ("all", {"all"})
    SINGLE_TURN = ("single_turn", {"single_turn"})
    MULTI_TURN = ("multi_turn", {"multi_turn"})

    # Strategies for tweaking jailbreak efficacy through attack patterns
    ManyShot = ("many_shot", {"single_turn"})
    PromptSending = ("prompt_sending", {"single_turn"})
    Crescendo = ("crescendo", {"multi_turn"})
    RedTeaming = ("red_teaming", {"multi_turn"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """
        Get the set of tags that represent aggregate categories.

        Returns:
            set[str]: Set of tags that are aggregate markers.
        """
        # Include base class aggregates ("all") and add scenario-specific ones
        return super().get_aggregate_tags() | {"single_turn", "multi_turn"}


class Jailbreak(Scenario):
    """
    Jailbreak scenario implementation for PyRIT.

    This scenario tests how vulnerable models are to jailbreak attacks by applying
    various single-turn jailbreak templates to a set of test prompts. The responses are
    scored to determine if the jailbreak was successful.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            type[ScenarioStrategy]: The JailbreakStrategy enum class.
        """
        return JailbreakStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: JailbreakStrategy.ALL.
        """
        return JailbreakStrategy.ALL

    @classmethod
    def required_datasets(cls) -> list[str]:
        """Return a list of dataset names required by this scenario."""
        return ["airt_harms"]

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """
        Return the default dataset configuration for this scenario.

        Returns:
            DatasetConfiguration: Configuration with airt_harms dataset.
        """
        return DatasetConfiguration(dataset_names=["airt_harms"], max_dataset_size=4)

    @apply_defaults
    def __init__(
        self,
        *,
        objective_scorer: Optional[TrueFalseScorer] = None,
        include_baseline: bool = False,
        scenario_result_id: Optional[str] = None,
        k: Optional[int] = None,
        n: int = 1,
        jailbreaks: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the jailbreak scenario.

        Args:
            objective_scorer (Optional[TrueFalseScorer]): Scorer for detecting successful jailbreaks
                (non-refusal). If not provided, defaults to an inverted refusal scorer.
            include_baseline (bool): Whether to include a baseline atomic attack that sends all
                objectives without modifications. Defaults to True.
            scenario_result_id (Optional[str]): Optional ID of an existing scenario result to resume.
            k (Optional[int]): Choose k random jailbreaks rather than using all of them.
            n (Optional[int]): Number of times to try each jailbreak.
            jailbreaks (Optional[int]): Dedicated list of jailbreaks to run.

        Raises:
            ValueError: If both jailbreaks and k are provided, as random selection
                is incompatible with a predetermined list.

        """
        if jailbreaks and k:
            raise ValueError("Please provide only one of `k` (random selection) or `jailbreaks` (specific selection).")

        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)

        self._k = k
        self._n = n

        if jailbreaks:
            self._validate_jailbreaks_subset(jailbreaks)
            self._jailbreaks = jailbreaks
        else:
            self._jailbreaks = TextJailBreak.get_all_jailbreak_templates()

        super().__init__(
            name="Jailbreak",
            version=self.version,
            strategy_class=JailbreakStrategy,
            objective_scorer=objective_scorer,
            include_default_baseline=include_baseline,
            scenario_result_id=scenario_result_id,
        )

        # Will be resolved in _get_atomic_attacks_async
        self._seed_groups: Optional[List[SeedAttackGroup]] = None

    def _validate_jailbreaks_subset(self, jailbreaks: List[str]) -> None:
        """
        Validate that the provided jailbreaks exist before moving on with initialization.

        Args:
            jailbreaks (List[str]): List of jailbreak names.

        Raises:
            ValueError: If jailbreaks not discovered.
        """
        all_templates = TextJailBreak.get_all_jailbreak_templates()
        diff = set(jailbreaks) - set(all_templates)
        if len(diff) > 0:
            raise ValueError(f"Error: could not find templates `{diff}`!")

    def _get_default_objective_scorer(self) -> TrueFalseScorer:
        """
        Retrieve the default objective scorer.

        The scoring mechanism detects non-refusal: if the target model refuses to
        comply with the jailbreak attempt, we want that to register as a negative score.
        A successful jailbreak (non-refusal) scores as True.

        Returns:
            TrueFalseScorer: A scorer that returns True when the model does NOT refuse.
        """
        refusal_scorer = TrueFalseInverterScorer(
            scorer=SelfAskRefusalScorer(
                chat_target=OpenAIChatTarget(
                    endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
                    model_name=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL"),
                )
            )
        )
        return refusal_scorer

    def _resolve_seed_groups(self) -> List[SeedAttackGroup]:
        """
        Resolve seed groups from dataset configuration.

        Returns:
            List[SeedAttackGroup]: List of seed attack groups with objectives to be tested.
        """
        # Use dataset_config (guaranteed to be set by initialize_async)
        seed_groups = self._dataset_config.get_all_seed_attack_groups()

        if not seed_groups:
            self._raise_dataset_exception()

        return list(seed_groups)

    def _get_all_jailbreak_templates(self) -> List[str]:
        """
        Retrieve all available jailbreak templates.

        Returns:
            List[str]: List of jailbreak template file names.
        """
        if not self._k:
            return TextJailBreak.get_all_jailbreak_templates()
        else:
            return TextJailBreak.get_all_jailbreak_templates(k=self._k)

    async def _get_atomic_attack_from_strategy_async(
        self, *, strategy: str, jailbreak_template_name: str
    ) -> AtomicAttack:
        """
        Create an atomic attack for a specific jailbreak template.

        Args:
            strategy (str): JailbreakStrategy to use.
            jailbreak_template_name (str): Name of the jailbreak template file.

        Returns:
            AtomicAttack: An atomic attack using the specified jailbreak template.

        Raises:
            ValueError: If an invalid strategy is provided.
        """
        # objective_target is guaranteed to be non-None by parent class validation
        assert self._objective_target is not None

        # Create the jailbreak converter
        jailbreak_converter = TextJailbreakConverter(
            jailbreak_template=TextJailBreak(template_file_name=jailbreak_template_name)
        )

        # Create converter configuration
        converter_config = AttackConverterConfig(
            request_converters=PromptConverterConfiguration.from_converters(converters=[jailbreak_converter])
        )

        attack = AttackStrategy
        match strategy:
            case "many_shot":
                attack = ManyShotJailbreakAttack
            case "prompt_sending":
                attack = PromptSendingAttack
            case "crescendo":
                attack = CrescendoAttack
            case "red_teaming":
                attack = RedTeamingAttack
            case _:
                raise ValueError(f"Unknown JailbreakStrategy `{strategy}`.")

        # Create the attack
        attack = attack(
            objective_target=self._objective_target,
            attack_scoring_config=self._scorer_config,
            attack_converter_config=converter_config,
        )

        # Extract template name without extension for the atomic attack name
        template_name = Path(jailbreak_template_name).stem

        return AtomicAttack(
            atomic_attack_name=f"jailbreak_{template_name}", attack=attack, seed_groups=self._seed_groups
        )

    async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
        """
        Generate atomic attacks for each jailbreak template.

        This method creates an atomic attack for each retrieved jailbreak template.

        Returns:
            List[AtomicAttack]: List of atomic attacks to execute, one per jailbreak template.

        Raises:
            ValueError: If self._jailbreaks is not a subset of all jailbreak templates.
        """
        atomic_attacks: List[AtomicAttack] = []

        # Retrieve seed prompts based on selected strategies
        self._seed_groups = self._resolve_seed_groups()

        strategies = ScenarioCompositeStrategy.extract_single_strategy_values(
            composites=self._scenario_composites, strategy_type=JailbreakStrategy
        )

        for strategy in strategies:
            for template_name in self._jailbreaks:
                atomic_attack = await self._get_atomic_attack_from_strategy_async(
                    strategy=strategy, jailbreak_template_name=template_name
                )
                atomic_attacks.extend([atomic_attack] * self._n)

        return atomic_attacks
