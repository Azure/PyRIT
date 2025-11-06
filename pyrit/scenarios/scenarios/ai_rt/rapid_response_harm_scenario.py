
from inspect import signature
import os
from typing import Dict, List, Optional, Sequence, Type, TypeVar

from pyrit.common.apply_defaults import apply_defaults
from pyrit.executor.attack.core.attack_config import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack import AttackStrategy, CrescendoAttack, MultiPromptSendingAttack, PromptSendingAttack
from pyrit.memory.central_memory import CentralMemory
from pyrit.models.seed_group import SeedGroup
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget
from pyrit.scenarios.atomic_attack import AtomicAttack
from pyrit.scenarios.scenario import Scenario
from pyrit.scenarios.scenario_strategy import ScenarioCompositeStrategy, ScenarioStrategy
from pyrit.score import (
    SelfAskRefusalScorer,
    TrueFalseInverterScorer,
    TrueFalseScorer,
)

AttackStrategyT = TypeVar("AttackStrategyT", bound=AttackStrategy)


class RapidResponseHarmStrategy(ScenarioStrategy):
    ALL = ("all", {"all"})
    HATE = ("hate", {"hate"})
    FAIRNESS = ("fairness", {"fairness"})
    VIOLENCE = ("violence", {"violence"})
    SEXUAL = ("sexual", {"sexual"})
    HARASSMENT = ("harassment", {"harassment"})
    MISINFORMATION = ("misinformation", {"misinformation"})
    LEAKAGE = ("leakage", {"leakage"})

    # hate strategies
    HateHighSchoolDebate = ("hate_high_school_debates", {"hate", "harm"})
    HateCrimeDirect = ("hate_crime_direct", {"hate", "harm"})
    HateFictionalStory = ("hate_fictional_story", {"hate", "harm"})
    HateHistoricalEvents = ("hate_historical_events", {"hate", "harm"})

    # fairness strategies
    FairnessPronounInference = ("fairness_pronoun_inference", {"fairness", "harm"})
    FairnessEthnicityInference = ("fairness_ethnicity_inference", {"fairness", "harm"})
    FairnessYesNoQuestions = ("fairness_yes_no_questions", {"fairness", "harm"})
    FairnessBias = ("fairness_bias", {"fairness", "harm"})

    # violence strategies
    ViolenceCivic = ("violence_civic", {"violence", "harm"})
    ViolenceProtestDisruption = ("violence_protest_disruption", {"violence", "harm"})

    # sexual strategies
    SexualExplicitContent = ("sexual_explicit_content", {"sexual", "harm"})
    SexualVocabulary = ("sexual_vocabulary", {"sexual", "harm"})
    
    # harassment strategies
    HarassmentBullying = ("harassment_bullying", {"harassment", "harm"})
    HarassmentUpskilling = ("harassment_upskilling", {"harassment", "harm"})
    HarassmentFictionalStory = ("harassment_fictional_story", {"harassment", "harm"})
    
    # misinformation strategies
    MisinformationElections = ("misinformation_elections", {"misinformation", "harm"})
    MisinformationFictionalStory = ("misinformation_fictional_story", {"misinformation", "harm"})
    # leakage strategies
    LeakageBookContent = ("leakage_book_content", {"leakage", "harm"})

    # multi-turn attack strategies
    MultiTurn = ("multi_turn", {"attack"})
    Crescendo = ("crescendo", {"attack"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """
        Get the set of tags that represent aggregate categories.

        Returns:
            set[str]: Set of tags that are aggregate markers.
        """
        # Include base class aggregates ("all") and add harm-specific ones
        return super().get_aggregate_tags() | {"hate", "fairness", "violence", "sexual", "harassment", "misinformation", "leakage"}
    
    @classmethod
    def supports_composition(cls) -> bool:
        """
        Indicate that RapidResponseHarmStrategy supports composition.

        Returns:
            bool: True, as RapidResponseHarm strategies can be composed together (with rules).
        """
        return True

    @classmethod
    def validate_composition(cls, strategies: Sequence[ScenarioStrategy]) -> None:
        """
        Validate whether the given RapidResponseHarm strategies can be composed together.

        RapidResponseHarm-specific composition rules:
        - Multiple attack strategies (e.g., Crescendo, MultiTurn) cannot be composed together
        - Mutliple harm strategies can be composed together
        - At most one attack can be composed of one harm

        Args:
            strategies (Sequence[ScenarioStrategy]): The strategies to validate for composition.

        Raises:
            ValueError: If the composition violates Foundry's rules (e.g., multiple attack).
        """
        if not strategies:
            raise ValueError("Cannot validate empty strategy list")

        # Filter to only RapidResponseHarmStrategy instances
        rapid_response_harm_strategies = [s for s in strategies if isinstance(s, RapidResponseHarmStrategy)]

        # Cannot compose multiple attack strategies
        attacks = [s for s in rapid_response_harm_strategies if "attack" in s.tags]
        harms = [s for s in rapid_response_harm_strategies if "harm" in s.tags]

        if len(attacks) > 1:
            raise ValueError(
                f"Cannot compose multiple attack strategies together: {[a.value for a in attacks]}. "
                f"Only one attack strategy is allowed per composition."
            )
        if len(harms) > 1
            raise ValueError(
                f"Cannot compose multiple harm strategies together: {[h.value for h in harms]}. "
                f"Only one harm strategy is allowed per composition."
            )


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
            Type[ScenarioStrategy]: The FoundryStrategy enum class.
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
        objective_target: PromptTarget,
        scenario_strategies: Sequence[RapidResponseHarmStrategy | ScenarioCompositeStrategy] | None = None,
        adversarial_chat: Optional[PromptChatTarget] = None,
        objective_scorer: Optional[TrueFalseScorer] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        max_concurrency: int = 5,
        converters: Optional[List[PromptConverter]] = None,
        objective_dataset_path: Optional[str] = None,
    ):
        """
        Initialize the HarmScenario.

        Args:
            objective_target (PromptTarget): The target model to test for harms vulnerabilities.
            scenario_strategies (Sequence[HarmStrategy | ScenarioCompositeStrategy] | None):
                The harm strategies or composite strategies to include in this scenario. If None, defaults to HarmStrategy.ALL.
            adversarial_chat (Optional[PromptChatTarget]): The chat target used for adversarial multi
            objective_scorer (Optional[TrueFalseScorer]): The scorer used to evaluate if the model
                successfully decoded the payload. Defaults to DecodingScorer with encoding_scenario
                category.
            memory_labels (Optional[Dict[str, str]]): Optional labels to attach to memory entries
                for tracking and filtering.
            max_concurrency (int): Maximum number of concurrent operations. Defaults to 5.
            converters (Optional[List[PromptConverter]]): List of converters to apply to the attack prompts.
            objective_dataset_path (Optional[str]): Path of the dataset to use for the objectives. If None,
                use objectives defined in rapid_response_harm_objectives which should be pre-loaded into memory.
        """

        self._objective_target = objective_target
        self._adversarial_chat = adversarial_chat if adversarial_chat else self._get_default_adversarial_target()
        self._objective_scorer = objective_scorer if objective_scorer else self._get_default_scorer()
        self._converters = converters if converters else None
        self.objective_dataset_path = objective_dataset_path if objective_dataset_path else "rapid_response_harm_objectives"
        self._objective_scorer_config = AttackScoringConfig(objective_scorer=self._objective_scorer)
        self._attack_adversarial_config = AttackAdversarialConfig(target=self._adversarial_chat)

        self._memory_labels = memory_labels or {}

        self._rapid_response_harm_strategy_compositiion = RapidResponseHarmStrategy.prepare_scenario_strategies(scenario_strategies,
                                                                                    default_aggregate=RapidResponseHarmStrategy.ALL)

        super().__init__(
            name="Rapid Response Harm Scenario",
            version=self.version,
            memory_labels=memory_labels,
            max_concurrency=max_concurrency,
            objective_scorer_identifier=self._objective_scorer.get_identifier(),
            objective_target=objective_target,
        )

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
        Retrieve the list of AtomicAttack instances in this scenario.

        Returns:
            List[AtomicAttack]: The list of AtomicAttack instances in this scenario.
        """
        return self._get_rapid_response_harm_attacks()

    
    def _get_rapid_response_harm_attacks(self) -> List[AtomicAttack]:
        """
        Retrieve the list of AtomicAttack instances for harm strategies.

        Returns:
            List[AtomicAttack]: The list of AtomicAttack instances for harm strategies.
        """
        atomic_attacks: List[AtomicAttack] = []
        for strategy in self._rapid_response_harm_strategy_compositiion:
            atomic_attacks.append(self._get_attack_from_strategy(composite_strategy=strategy))
        return atomic_attacks

    def _get_attack_from_strategy(
        self,
        composite_strategy: ScenarioCompositeStrategy,
    ) -> AtomicAttack:
        """
        Create an AtomicAttack instance based on the provided strategy.

        Args:
            strategy (ScenarioStrategy): The strategy to create the attack from.

        Returns:
            AtomicAttack: The constructed AtomicAttack instance.
        """

        attack: AttackStrategy

        # Extract RapidResponseHarmStrategy enums from the composite
        strategy_list = [s for s in composite_strategy.strategies if isinstance(s, RapidResponseHarmStrategy)]


        # Determine the attack type based on the strategy tags
        attack_type: type[AttackStrategy] = PromptSendingAttack
        attack_tag = [s for s in strategy_list if "attack" in s.tags]
        attack_type: type[AttackStrategy] = PromptSendingAttack
        if attack_tag:
            if attack_tag[0] == RapidResponseHarmStrategy.Crescendo:
                attack_type = CrescendoAttack
            elif attack_tag[0] == RapidResponseHarmStrategy.MultiTurn:
                attack_type = MultiPromptSendingAttack
            else:
                raise ValueError(f"Unknown attack strategy: {attack_tag[0].value}")
            

        attack = self._get_attack(attack_type=attack_type)

        harm_tag = [s for s in strategy_list if "harm" in s.tags]
        if not harm_tag:
            raise ValueError(f"No harm strategy found in composition: {[s.value for s in strategy_list]}")
        if harm_tag[0].value not in RapidResponseHarmStrategy.get_all_strategies():
            raise ValueError(f"Unknown harm strategy: {harm_tag[0].value}")
        
        # Retrieve objectives from CentralMemory based on harm tag
        memory = CentralMemory.get_memory_instance()
        harm_dataset_name = f"{self.objective_dataset_path}_{harm_tag[0].value}"
        seed_groups = memory.get_seed_groups(dataset_name=harm_dataset_name)
        strategy_objectives: list[str]= [obj.objective.value for obj in seed_groups if obj.objective is not None]
        if len(strategy_objectives) == 0:
            raise ValueError(f"No objectives found in the dataset {harm_dataset_name}. Ensure that the dataset is properly loaded into CentralMemory.")

        return AtomicAttack(
            atomic_attack_name=composite_strategy.name,
            attack=attack,
            objectives=strategy_objectives,
            memory_labels=self._memory_labels,
        )

    def _get_attack(
        self,
        *,
        attack_type: type[AttackStrategyT],
    ) -> AttackStrategyT:
        """
        Create an attack instance with the specified converters.

        This method creates an instance of an AttackStrategy subclass with the provided
        converters configured as request converters. For multi-turn attacks that require
        an adversarial target (e.g., CrescendoAttack), the method automatically creates
        an AttackAdversarialConfig using self._adversarial_chat.

        Supported attack types include:
        - PromptSendingAttack (single-turn): Only requires objective_target and attack_converter_config
        - CrescendoAttack (multi-turn): Also requires attack_adversarial_config (auto-generated)
        - RedTeamingAttack (multi-turn): Also requires attack_adversarial_config (auto-generated)
        - Other attacks with compatible constructors

        Args:
            attack_type (type[AttackStrategyT]): The attack strategy class to instantiate.
                Must accept objective_target and attack_converter_config parameters.
            converters (list[PromptConverter]): List of converters to apply as request converters.

        Returns:
            AttackStrategyT: An instance of the specified attack type with configured converters.

        Raises:
            ValueError: If the attack requires an adversarial target but self._adversarial_chat is None.
        """
        attack_converter_config: Optional[AttackConverterConfig] = None
        if self._converters is not  None:
            attack_converter_config = AttackConverterConfig(
                request_converters=PromptConverterConfiguration.from_converters(converters=self._converters)
            )

        # Build kwargs with required parameters
        kwargs = {
            "objective_target": self._objective_target,
            "attack_converter_config": attack_converter_config,
            "attack_scoring_config": AttackScoringConfig(objective_scorer=self._objective_scorer),
        }

        # Check if the attack type requires attack_adversarial_config by inspecting its __init__ signature
        sig = signature(attack_type.__init__)
        if "attack_adversarial_config" in sig.parameters:
            # This attack requires an adversarial config
            if self._adversarial_chat is None:
                raise ValueError(
                    f"{attack_type.__name__} requires an adversarial target, "
                    f"but self._adversarial_chat is None. "
                    f"Please provide adversarial_chat when initializing {self.__class__.__name__}."
                )

            # Create the adversarial config from self._adversarial_target
            attack_adversarial_config = AttackAdversarialConfig(target=self._adversarial_chat)
            kwargs["attack_adversarial_config"] = attack_adversarial_config

        # Type ignore is used because this is a factory method that works with compatible
        # attack types. The caller is responsible for ensuring the attack type accepts
        # these constructor parameters.
        return attack_type(**kwargs)  # type: ignore[arg-type, call-arg]
