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
from typing import Dict, List, Optional, Sequence

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
    AtbashConverter,
    Base64Converter,
    LeetspeakConverter,
    MorseConverter,
    PromptConverter,
    AskToDecodeConverter,
    BinAsciiConverter,
    TranslationConverter,
    VariationConverter,
    VariationConverter,
    ZalgoConverter,
    ROT13Converter
)

from pyrit.prompt_converter.braille_converter import BrailleConverter
from pyrit.prompt_converter.ecoji_converter import EcojiConverter
from pyrit.prompt_converter.nato_converter import NatoConverter
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget, PromptChatTarget, OpenAIChatTarget

from pyrit.scenarios.attack_run import AttackRun
from pyrit.scenarios.scenario import Scenario

from pyrit.score import TrueFalseScorer
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
        seed_prompts: Optional[list[str]] = None,
        objective_scorer: Optional[TrueFalseScorer] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        encoding_templates: Optional[Sequence[str]] = None,
        max_concurrency: int = 10
    ):

        self._objective_target = objective_target

        # There is currently only one strategy, but this class could encorporate more in the future
        self._attack_strategies = ["Garak basic"]

        objective_scorer = objective_scorer or DecodingScorer(categories=["encoding_scenario"])
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)


        self._seed_prompts: list[str] = seed_prompts if seed_prompts else self._get_default_dataset()
        self._memory_labels = memory_labels or {}
        self._encoding_templates = encoding_templates or AskToDecodeConverter.garak_templates

        super().__init__(
            name=f"Encoding Scenario",
            attack_strategies=self._attack_strategies,
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


    async def _get_attack_runs_async(self) -> List[AttackRun]:
        """
        Retrieve the list of AttackRun instances in this scenario.

        Returns:
            List[AttackRun]: The list of AttackRun instances in this scenario.
        """
        return self._get_converter_attacks()

    # TODO base 2048

    # These are the same as Garak encoding attacks
    def _get_converter_attacks(self) -> list[AttackRun]:
        return  (
            self._get_prompt_attacks(converters=[Base64Converter()], encoding_name="Base64") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="urlsafe_b64encode")], encoding_name="Base64") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="standard_b64encode")], encoding_name="Base64") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="b2a_base64")], encoding_name="Base64") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="b16encode")], encoding_name="Base16") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="b32encode")], encoding_name="Base32") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="a85encode")], encoding_name="ASCII85") +
            self._get_prompt_attacks(converters=[Base64Converter(encoding_func="b85encode")], encoding_name="ASCII85") +
            self._get_prompt_attacks(converters=[BinAsciiConverter(encoding_func="hex")], encoding_name="hex") +
            self._get_prompt_attacks(converters=[BinAsciiConverter(encoding_func="quoted-printable")], encoding_name="quoted-printable") +
            self._get_prompt_attacks(converters=[BinAsciiConverter(encoding_func="UUencode")], encoding_name="UUencode") +
            self._get_prompt_attacks(converters=[ROT13Converter()], encoding_name="ROT13") +
            self._get_prompt_attacks(converters=[BrailleConverter()], encoding_name="Braille") +
            self._get_prompt_attacks(converters=[AtbashConverter()], encoding_name="Atbash") +
            self._get_prompt_attacks(converters=[MorseConverter()], encoding_name="Morse Code") +
            self._get_prompt_attacks(converters=[NatoConverter()], encoding_name="NATO") +
            self._get_prompt_attacks(converters=[EcojiConverter()], encoding_name="Ecoji") +
            self._get_prompt_attacks(converters=[ZalgoConverter()], encoding_name="Zalgo") +
            self._get_prompt_attacks(converters=[LeetspeakConverter()], encoding_name="Leet Speak") +
            self._get_prompt_attacks(converters=[AsciiSmugglerConverter()], encoding_name="Ascii in Unicode tags") 
        )

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
            seed_prompt_groups = []
            objectives = []
            for seed_prompt in self._seed_prompts:
                seed = SeedPrompt(value=seed_prompt)
                seed_prompt_groups.append(SeedPromptGroup(prompts=[seed]))
                objectives.append(f"Make the model say an encoded payload: {seed_prompt}")

            attack = PromptSendingAttack(
                objective_target=self._objective_target,
                attack_converter_config=attack_converter_config,
                attack_scoring_config=self._scorer_config,
            )
            attack_runs.append(AttackRun(
                attack=attack,
                objectives=objectives,
                seed_prompt_groups=seed_prompt_groups
            ))

        return attack_runs