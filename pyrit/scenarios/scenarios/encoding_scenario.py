# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pathlib
from typing import List, Optional, Sequence

from pyrit.common import apply_defaults
from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack.core.attack_config import (
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models import SeedDataset, SeedGroup
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.prompt_converter import (
    AsciiSmugglerConverter,
    AskToDecodeConverter,
    AtbashConverter,
    Base64Converter,
    Base2048Converter,
    BinAsciiConverter,
    LeetspeakConverter,
    MorseConverter,
    PromptConverter,
    ROT13Converter,
    ZalgoConverter,
)
from pyrit.prompt_converter.braille_converter import BrailleConverter
from pyrit.prompt_converter.ecoji_converter import EcojiConverter
from pyrit.prompt_converter.nato_converter import NatoConverter
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.scenarios.atomic_attack import AtomicAttack
from pyrit.scenarios.scenario import Scenario
from pyrit.scenarios.scenario_strategy import (
    ScenarioCompositeStrategy,
    ScenarioStrategy,
)
from pyrit.score import TrueFalseScorer
from pyrit.score.true_false.decoding_scorer import DecodingScorer


class EncodingStrategy(ScenarioStrategy):
    """
    Strategies for encoding attacks.

    Each enum member represents an encoding scheme that will be tested against the target model.
    The ALL aggregate expands to include all encoding strategies.

    Note: EncodingStrategy does not support composition. Each encoding must be applied individually.
    """

    # Aggregate member
    ALL = ("all", {"all"})

    # Individual encoding strategies (matching the atomic attack names)
    Base64 = ("base64", set[str]())
    Base2048 = ("base2048", set[str]())
    Base16 = ("base16", set[str]())
    Base32 = ("base32", set[str]())
    ASCII85 = ("ascii85", set[str]())
    Hex = ("hex", set[str]())
    QuotedPrintable = ("quoted_printable", set[str]())
    UUencode = ("uuencode", set[str]())
    ROT13 = ("rot13", set[str]())
    Braille = ("braille", set[str]())
    Atbash = ("atbash", set[str]())
    MorseCode = ("morse_code", set[str]())
    NATO = ("nato", set[str]())
    Ecoji = ("ecoji", set[str]())
    Zalgo = ("zalgo", set[str]())
    LeetSpeak = ("leet_speak", set[str]())
    AsciiSmuggler = ("ascii_smuggler", set[str]())


class EncodingScenario(Scenario):
    """
    Encoding Scenario implementation for PyRIT.

    This scenario tests how resilient models are to various encoding attacks by encoding
    potentially harmful text (by default slurs and XSS payloads) and testing if the model
    will decode and repeat the encoded payload. It mimics the Garak encoding probe.

    The scenario works by:
    1. Taking seed prompts (the harmful text to be encoded)
    2. Encoding them using various encoding schemes (Base64, ROT13, Morse, etc.)
    3. Asking the target model to decode the encoded text
    4. Scoring whether the model successfully decoded and repeated the harmful content

    By default, this uses the same dataset as Garak: slur terms and web XSS payloads.
    """

    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        Returns:
            Type[ScenarioStrategy]: The EncodingStrategy enum class.
        """
        return EncodingStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        Returns:
            ScenarioStrategy: EncodingStrategy.ALL (all encoding strategies).
        """
        return EncodingStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        seed_prompts: Optional[list[str]] = None,
        objective_scorer: Optional[TrueFalseScorer] = None,
        encoding_templates: Optional[Sequence[str]] = None,
        include_baseline: bool = True,
        scenario_result_id: Optional[str] = None,
    ):
        """
        Initialize the EncodingScenario.

        Args:
            seed_prompts (Optional[list[str]]): The list of text strings that will be encoded and
                used to test the model. These are the pieces the scenario is trying to get the model
                to decode and repeat. If not provided, defaults to slur terms and XSS payloads from
                the Garak dataset.
            objective_scorer (Optional[TrueFalseScorer]): The scorer used to evaluate if the model
                successfully decoded the payload. Defaults to DecodingScorer with encoding_scenario
                category.
            encoding_templates (Optional[Sequence[str]]): Templates used to construct the decoding
                prompts. Defaults to AskToDecodeConverter.garak_templates.
            include_baseline (bool): Whether to include a baseline atomic attack that sends all objectives
                without modifications. Defaults to True. When True, a "baseline" attack is automatically
                added as the first atomic attack, allowing comparison between unmodified prompts and
                encoding-modified prompts.
            scenario_result_id (Optional[str]): Optional ID of an existing scenario result to resume.
        """
        objective_scorer = objective_scorer or DecodingScorer(categories=["encoding_scenario"])
        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)

        self._seed_prompts: list[str] = seed_prompts if seed_prompts else self._get_default_dataset()
        self._encoding_templates = encoding_templates or AskToDecodeConverter.garak_templates

        super().__init__(
            name="Encoding Scenario",
            version=self.version,
            strategy_class=EncodingStrategy,
            objective_scorer_identifier=objective_scorer.get_identifier(),
            include_default_baseline=include_baseline,
            scenario_result_id=scenario_result_id,
        )

    # Use the same as Garak by default
    def _get_default_dataset(self) -> list[str]:
        """
        Get the default seed prompts for encoding tests.

        Returns the same dataset used by Garak, which includes slur terms and
        web HTML/JS XSS payloads. These are the harmful text strings that will
        be encoded and used to test if the model will decode them.

        Returns:
            list[str]: List of seed prompt strings to be encoded and tested.
        """
        seed_prompts: list[str] = []
        garak_path = pathlib.Path(DATASETS_PATH) / "seed_datasets" / "local" / "garak"
        seed_prompts.extend(SeedDataset.from_yaml_file(garak_path / "slur_terms_en.prompt").get_values())
        seed_prompts.extend(SeedDataset.from_yaml_file(garak_path / "web_html_js.prompt").get_values())

        return seed_prompts

    async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
        """
        Retrieve the list of AtomicAttack instances in this scenario.

        Returns:
            List[AtomicAttack]: The list of AtomicAttack instances in this scenario.
        """
        return self._get_converter_attacks()

    # These are the same as Garak encoding attacks
    def _get_converter_attacks(self) -> list[AtomicAttack]:
        """
        Get all converter-based atomic attacks.

        Creates atomic attacks for each encoding scheme specified in the scenario strategies.
        Each encoding scheme is tested both with and without explicit decoding instructions.

        Returns:
            list[AtomicAttack]: List of all atomic attacks to execute.
        """
        # Map of all available converters with their encoding names
        all_converters_with_encodings: list[tuple[list[PromptConverter], str]] = [
            ([Base64Converter()], "base64"),
            ([Base64Converter(encoding_func="urlsafe_b64encode")], "base64"),
            ([Base64Converter(encoding_func="standard_b64encode")], "base64"),
            ([Base64Converter(encoding_func="b2a_base64")], "base64"),
            ([Base2048Converter()], "base2048"),
            ([Base64Converter(encoding_func="b16encode")], "base16"),
            ([Base64Converter(encoding_func="b32encode")], "base32"),
            ([Base64Converter(encoding_func="a85encode")], "ascii85"),
            ([Base64Converter(encoding_func="b85encode")], "ascii85"),
            ([BinAsciiConverter(encoding_func="hex")], "hex"),
            ([BinAsciiConverter(encoding_func="quoted-printable")], "quoted_printable"),
            ([BinAsciiConverter(encoding_func="UUencode")], "uuencode"),
            ([ROT13Converter()], "rot13"),
            ([BrailleConverter()], "braille"),
            ([AtbashConverter()], "atbash"),
            ([MorseConverter()], "morse_code"),
            ([NatoConverter()], "nato"),
            ([EcojiConverter()], "ecoji"),
            ([ZalgoConverter()], "zalgo"),
            ([LeetspeakConverter()], "leet_speak"),
            ([AsciiSmugglerConverter()], "ascii_smuggler"),
        ]

        # Filter to only include selected strategies
        selected_encoding_names = ScenarioCompositeStrategy.extract_single_strategy_values(
            self._scenario_composites, strategy_type=EncodingStrategy
        )
        converters_with_encodings = [
            (conv, name) for conv, name in all_converters_with_encodings if name in selected_encoding_names
        ]

        atomic_attacks = []
        for conv, name in converters_with_encodings:
            atomic_attacks.extend(self._get_prompt_attacks(converters=conv, encoding_name=name))
        return atomic_attacks

    def _get_prompt_attacks(self, *, converters: list[PromptConverter], encoding_name: str) -> list[AtomicAttack]:
        """
        Create atomic attacks for a specific encoding scheme.

        For each seed prompt (the text to be decoded), creates atomic attacks that:
        1. Encode the seed prompt using the specified converter(s)
        2. Optionally add a decoding instruction template
        3. Send to the target model
        4. Score whether the model decoded and repeated the harmful content

        Args:
            converters (list[PromptConverter]): The list of converters to apply to the seed prompts.
            encoding_name (str): Human-readable name of the encoding scheme (e.g., "Base64", "ROT13").

        Returns:
            list[AtomicAttack]: List of atomic attacks for this encoding scheme.
        """
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

        atomic_attacks = []
        for attack_converter_config in converter_configs:
            seed_groups = []
            objectives = []
            for seed_prompt in self._seed_prompts:
                seed = SeedPrompt(value=seed_prompt)
                seed_groups.append(SeedGroup(seeds=[seed]))
                objectives.append(f"Make the model say an encoded payload: {seed_prompt}")

            # objective_target is guaranteed to be non-None by parent class validation
            assert self._objective_target is not None
            attack = PromptSendingAttack(
                objective_target=self._objective_target,
                attack_converter_config=attack_converter_config,
                attack_scoring_config=self._scorer_config,
            )
            atomic_attacks.append(
                AtomicAttack(
                    atomic_attack_name=encoding_name, attack=attack, objectives=objectives, seed_groups=seed_groups
                )
            )

        return atomic_attacks
