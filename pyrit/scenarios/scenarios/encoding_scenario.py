# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Encoding Scenario implementation for PyRIT.

Mimics Garak encoding probe.
A scenarios to test how resilient models are to various encoding attacks. 
Basically encodes text (by default slurs and xss) and sees if the model will say the encoded payload.
"""

import asyncio
import os
from enum import Enum
import pathlib
import random
from typing import Dict, Optional, Sequence

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets.harmbench_dataset import fetch_harmbench_dataset
from pyrit.executor.attack.core.attack_config import (
    AttackConverterConfig,
    AttackScoringConfig,
)

from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.models.seed_prompt_dataset import SeedPromptDataset
from pyrit.models.seed_prompt_group import SeedPromptGroup
from pyrit.prompt_converter import (
    AsciiSmugglerConverter,
    AsciiArtConverter,
    AtbashConverter,
    Base64Converter,
    LeetspeakConverter,
    MorseConverter,
    PromptConverter,
    UnicodeConfusableConverter,
    UnicodeSubstitutionConverter,
    VariationSelectorSmugglerConverter,
    SneakyBitsSmugglerConverter,
    BinaryConverter,
    AskToDecodeConverter,
    SuperscriptConverter,
    TextToHexConverter,
    ZeroWidthConverter,
    CodeChameleonConverter,
    TranslationConverter,
    VariationConverter,
    VariationConverter,
    ZalgoConverter
)

from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget, PromptChatTarget, OpenAIChatTarget

from pyrit.scenarios.attack_run import AttackRun
from pyrit.scenarios.scenario import Scenario

from pyrit.score import (
    AzureContentFilterScorer,
    FloatScaleThresholdScorer,
    SelfAskRefusalScorer,
    TrueFalseCompositeScorer,
    TrueFalseInverterScorer,
    TrueFalseScoreAggregator,
    TrueFalseScorer,
    SelfAskScaleScorer
)
from pyrit.score.true_false.decoding_scorer import DecodingScorer

"""
FAST directly mimics Garak encoding strategies
    does not make use of an adversarial LLM, and SeedPrompts are directly encoded.
MEDIUM encoding strategies make use an adversarial LLM in converters and to modify SeedPrompts.
    It does this by rephrasing the SeedPrompt and then translating into a different language.
    It adds an additional scorer by default.
"""
class EncodingStrategy(Enum):
    FAST = "fast"
    MEDIUM = "medium"


class EncodingScenario(Scenario):

    version: int = 1

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        attack_strategies: EncodingStrategy = EncodingStrategy.FAST,
        adversarial_chat: Optional[PromptChatTarget] = None,
        seed_prompts: Optional[list[str]] = None,
        objective_scorer: Optional[TrueFalseScorer] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        encoding_templates: Optional[Sequence[str]] = None,
        max_concurrency: int = 1
    ):

        self._objective_target = objective_target
        self._attack_strategies = [attack_strategies.value]

        # This will change under certain attack scenarios
        self._language = "English"
        self._seed_prompt_groups: Optional[list[SeedPromptGroup]] = None

        # only necessary for moderate and difficult strategies
        self._adversarial_chat = adversarial_chat if adversarial_chat else self._get_default_adversarial_target()


        objective_scorer = objective_scorer or DecodingScorer(categories=["encoding_scenario"])
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)


        self._seed_prompts: list[str] = seed_prompts if seed_prompts else self._get_default_dataset()
        self._memory_labels = memory_labels or {}
        self._encoding_templates = encoding_templates or AskToDecodeConverter.all_templates

        super().__init__(
            name=f"Encoding Scenario",
            attack_strategies=[attack_strategies.value],
            version=self.version,
            memory_labels=memory_labels,
            max_concurrency=max_concurrency,
        )

    # Use the same as Garak by default
    def _get_default_dataset(self) -> list[str]:
        seed_prompts = []
        garak_path = pathlib.Path(DATASETS_PATH) / "seed_prompts" / "garak"
        seed_prompts.extend(SeedPromptDataset.from_yaml_file(garak_path / "slur_terms_en.prompt").get_values())
        seed_prompts.extend(SeedPromptDataset.from_yaml_file(garak_path / "web_html_js.prompt").get_values())

        return seed_prompts

    async def initialize_async(self) -> None:
        """
        Build the attack runs for this scenario.
        
        This method handles async operations needed to build attack runs,
        such as rephrasing or translating objectives for moderate strategies.
        
        Returns:
            List of AttackRun instances configured for the scenario's strategy.
            
        Example:
            >>> scenario = EncodingScenario(
            ...     objective_target=target,
            ...     attack_strategies=EncodingStrategy.MODERATE
            ... )
            >>> attack_runs = await scenario.build_attack_runs_async()
            >>> scenario.initialize(attack_runs)
            >>> results = await scenario.run_async()
        """
        if self._attack_strategies[0] == EncodingStrategy.FAST.value:
            self._attack_runs = self._get_fast_attacks()
        elif self._attack_strategies[0] == EncodingStrategy.MEDIUM.value:
            self._attack_runs = await self._get_medium_attacks()
        else:
            raise ValueError(f"Unsupported encoding strategy: {self._attack_strategies}")

    def _get_default_adversarial_target(self):
        return OpenAIChatTarget(
            endpoint=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY"),
            temperature=0.7,
        )

    def _get_converter_attacks(self) -> list[AttackRun]:
        return  self._get_prompt_attacks(converters=[], encoding_name="") + \
                self._get_prompt_attacks(converters=[AskToDecodeConverter()], encoding_name="")

    def _get_fast_attacks(self) -> list[AttackRun]:
        return self._get_converter_attacks()

    async def _get_medium_attacks(self) -> list[AttackRun]:
        """
        Build medium difficulty attack runs.
        """

        self._language = random.choice(["Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian", "Chinese", "Japanese", "Korean"])

        rephrase_converter = VariationConverter(converter_target=self._adversarial_chat)
        translation_converter = TranslationConverter(converter_target=self._adversarial_chat, language=self._language)

        # modify the list of seed prompts
        async def _update_seed_prompt_async(prompt: str) -> SeedPromptGroup:
            rephrased_value = await rephrase_converter.convert_async(prompt=prompt)
            translated_value = await translation_converter.convert_async(prompt=rephrased_value.output_text)
            return SeedPromptGroup(
                prompts=[SeedPrompt(
                    value=translated_value.output_text
                )]
            )

        self._seed_prompt_groups = await asyncio.gather(*[_update_seed_prompt_async(p) for p in self._seed_prompts])
        return self._get_converter_attacks()


    def _get_prompt_attacks(
        self,
        *,
        converters: list[PromptConverter],
        encoding_name: str
    ) -> list[AttackRun]:


        converter_configs = [
            AttackConverterConfig(
                request_converters=PromptConverterConfiguration.from_converters(converters=converters)
            )
        ]

        for decode_type in self._encoding_templates:
            converters_ = converters[:] + [AskToDecodeConverter(template=decode_type, encoding_name=encoding_name)]

            converter_configs.append(
                AttackConverterConfig(
                    request_converters=PromptConverterConfiguration.from_converters(converters=converters_)
                )
            )

        attack_runs = []
        for attack_converter_config in converter_configs:
            attack = PromptSendingAttack(
                objective_target=self._objective_target,
                attack_converter_config=attack_converter_config,
                attack_scoring_config=self._scorer_config,
            )
            attack_runs.append(AttackRun(
                attack=attack,
                objectives=self._seed_prompts,
                seed_prompt_groups=self._seed_prompt_groups
            ))

        return attack_runs