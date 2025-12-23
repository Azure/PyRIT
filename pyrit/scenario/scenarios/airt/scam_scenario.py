# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Optional

from pyrit.common import apply_defaults
from pyrit.common.path import SCORER_SEED_PROMPT_PATH
from pyrit.executor.attack import RolePlayAttack, RolePlayPaths
from pyrit.executor.attack.core.attack_config import (
    AttackAdversarialConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.core.attack_strategy import AttackStrategy
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
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


class ScamStrategy(ScenarioStrategy):
    """
    Strategies for the Scam (Impersonation) Scenario.
    """

    ALL = ("all", {"all"})
    ROLE_PLAY = ("roleplay", {"roleplay"})
    MULTI_TURN = ("multi_turn", {"multi_turn"})

    # Persuasion = ("persuasion", {"roleplay"})
    # Movie = ("movie", {"roleplay"})
    # Trivia = ("trivia", {"roleplay"})
    # VideoGame = ("video_game", {"roleplay"})
    # TranslationConverter Strategy - add (Russian language) converter to each seed prompt + add "respond in english"


class ScamScenario(Scenario):
    """
    ScamScenario is a preconfigured scenario which evaluates a model's ability
    to facilitate various kinds of scams.
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

    @classmethod
    def required_datasets(cls) -> list[str]:
        """Return a list of dataset names required by this scenario."""
        return ["airt_scams"]

    @apply_defaults
    def __init__(
        self,
        *,
        objectives: Optional[List[str]] = None,
        objective_scorer: Optional[TrueFalseCompositeScorer] = None,
        adversarial_chat: Optional[PromptChatTarget] = None,
        include_baseline: bool = True,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the ScamScenario.

        Args:
            objectives (Optional[List[str]]): List of objectives to test for scam-related harms.
            objective_scorer (Optional[TrueFalseCompositeScorer]): Custom scorer for objective
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

        self._adversarial_chat = adversarial_chat if adversarial_chat else self._get_default_adversarial_target()
        self._adversarial_config = AttackAdversarialConfig(target=self._adversarial_chat)

        # Call super().__init__() first to initialize self._memory
        super().__init__(
            name="Scam Scenario",
            version=self.version,
            strategy_class=ScamStrategy,
            objective_scorer_identifier=objective_scorer.get_identifier(),
            include_default_baseline=include_baseline,
            scenario_result_id=scenario_result_id,
        )

        # Now we can safely access self._memory
        self._objectives = objectives if objectives else self._get_default_objectives()

    def _get_default_objective_scorer(self) -> TrueFalseCompositeScorer:
        """
        Retrieve the default objective scorer.
        The scoring mechanism is as follows:
        1. Detect for objective achieved using a SelfAskTrueFalseScorer.
        2. Detect for NON-refusal as a backstop. (If the target model refuses to produce scam material, we want that
        to register as a negative score).

        Returns:
            TrueFalseCompositeScorer: Default objective scorer with backstop and response evaluation.
        """
        scam_evaluator = SelfAskTrueFalseScorer(
            chat_target=OpenAIChatTarget(
                endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
                model_name=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL"),
                temperature=0.9,
            ),
            true_false_question_path=SCORER_SEED_PROMPT_PATH / "true_false_question" / "scams.yaml",
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
            aggregator=TrueFalseScoreAggregator.AND, scorers=[scam_evaluator, backstop]
        )

    def _get_default_adversarial_target(self) -> OpenAIChatTarget:
        """
        Provide an OpenAI target for the role-play rephrasing step.

        Returns:
            OpenAIChatTarget: Target that supplies the persuasion script rephrasing.
        """
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT2"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY2"),
            model_name=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL2"),
            temperature=1.2,
        )

    def _get_default_objectives(self) -> List[str]:
        """
        Load the default scam (impersonation) seed prompts.

        Returns:
            List[str]: Prompts from the seed dataset.
        """
        seed_objectives = self._memory.get_seeds(dataset_name="airt_scams", is_objective=True)

        if not seed_objectives:
            self._raise_dataset_exception()

        return [seed.value for seed in seed_objectives]

    def _get_atomic_attack_from_strategy(self, strategy: str) -> List[AtomicAttack]:
        """
        Translate the strategies into actual AtomicAttacks.

        Args:
            strategy (ScenarioCompositeStrategy): The strategy to create the attack from.

        Returns:
            List[AtomicAttack]: Configured for the specified strategy.

        Raises:
            ValueError: If an unknown ScamStrategy is provided.
        """
        # objective_target is guaranteed to be non-None by parent class validation
        assert self._objective_target is not None
        attack_strategy: Optional[AttackStrategy] = None

        if strategy == "multi_turn":
            attack_strategy = RedTeamingAttack(
                objective_target=self._objective_target,
                attack_scoring_config=self._scorer_config,
                attack_adversarial_config=self._adversarial_config,
            )
        elif strategy == "roleplay":
            # TODO: Return multiple RolePlayAttacks for each role-play subtype (persuasion, movie, trivia, video game)
            attack_strategy = RolePlayAttack(
                objective_target=self._objective_target,
                adversarial_chat=self._adversarial_chat,
                role_play_definition_path=RolePlayPaths.PERSUASION_SCRIPT.value,
                attack_scoring_config=self._scorer_config,
            )
        else:
            raise ValueError(f"Unknown ScamStrategy: {strategy}")

        return [AtomicAttack(
            atomic_attack_name=f"scam_{strategy}",
            attack=attack_strategy,
            objectives=self._objectives,
            memory_labels=self._memory_labels,
        )]

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
            atomic_attacks.extend(self._get_atomic_attack_from_strategy(strategy))

        return atomic_attacks
