# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random

from importlib.util import find_spec

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

# TODO: consider adding textattack as PyRIT dependency?
try:
    from textattack.augmentation import Augmenter
    from textattack.constraints.pre_transformation import (
        RepeatModification,
        StopwordModification,
    )
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.transformations import (
        CompositeTransformation,
        WordSwapEmbedding,
        WordSwapHomoglyphSwap,
        WordSwapNeighboringCharacterSwap,
        WordSwapRandomCharacterDeletion,
        WordSwapRandomCharacterInsertion,
    )

    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextBuggerConverter(PromptConverter):
    """
    Converts text using TextBugger adversarial attack techniques.

    The TextBugger attack applies five types of transformations:
        - 1. INSERT: Inserts a space into words to deceive classifiers.
        - 2. DELETE: Deletes random characters (except first/last).
        - 3. SWAP: Swaps adjacent characters (except first/last).
        - 4. SUB-C: Replaces characters with visually similar characters (homoglyphs).
        - 5. SUB-W: Replaces words with semantically similar words using GloVe embeddings.

    Inspired by TextBugger attack implementation from Project Moonshot:
    https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/textbugger_attack.py

    Original paper: https://arxiv.org/abs/1812.05271

    Note:
        This converter may require internet access for initial download of necessary
        resources such as GloVe embeddings and Universal Sentence Encoder model.
    """

    def __init__(
        self,
        *,
        word_swap_ratio: float = 0.2,
        top_k: int = 5,
        max_transformations: int = 5,
        use_universal_sentence_encoder: bool = False,
        semantic_threshold: float = 0.8,
    ) -> None:
        """
        Initializes the converter with TextBugger parameters.

        Args:
            word_swap_ratio (float): Percentage of words in a prompt that should be changed.
                Default is 0.2 (20% of words). Range: 0.0 to 1.0.
            top_k (int): Number of top semantic word candidates from GloVe embedding.
                Default is 5. Higher values provide more word substitution options.
            max_transformations (int): Maximum number of transformed versions to generate.
                Default is 5. Higher values create more adversarial variations.
            use_universal_sentence_encoder (bool): Whether to use Universal Sentence Encoder.
            semantic_threshold (float): Threshold for Universal Sentence Encoder similarity.
                Default is 0.8. Higher values ensure transformed text stays more semantically similar.
                Range: 0.0 to 1.0 (1.0 = identical meaning, 0.0 = completely different).

        Raises:
            ImportError: If TextAttack framework is not installed.
            ValueError: If any parameter is out of valid range.
        """
        if not TEXTATTACK_AVAILABLE:
            raise ImportError(
                "TextAttack framework is required for TextBuggerConverter. Install it with: pip install textattack"
            )

        if not (0.0 <= word_swap_ratio <= 1.0):
            raise ValueError(f"word_swap_ratio must be between 0.0 and 1.0, got {word_swap_ratio}")
        if not (0.0 <= semantic_threshold <= 1.0):
            raise ValueError(f"semantic_threshold must be between 0.0 and 1.0, got {semantic_threshold}")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if max_transformations < 1:
            raise ValueError(f"max_transformations must be at least 1, got {max_transformations}")

        self._word_swap_ratio = word_swap_ratio
        self._top_k = top_k
        self._max_transformations = max_transformations
        self._use_universal_sentence_encoder = use_universal_sentence_encoder
        self._semantic_threshold = semantic_threshold

        if self._use_universal_sentence_encoder:
            if find_spec("tensorflow_hub") is None:
                raise ImportError(
                    "Universal Sentence Encoder requires tensorflow_hub. Install it with: pip install tensorflow-hub"
                )

        # Augmenter instance that applies TextBugger transformations
        self._augmenter = self._create_augmenter()

    def _create_augmenter(self) -> "Augmenter":
        """
        Creates and configures the TextAttack augmenter with TextBugger transformations.

        Returns:
            Augmenter: Configured TextAttack augmenter instance.
        """
        # The five TEXTBUGGER transformations as defined in the paper
        transformation = CompositeTransformation(  # type: ignore
            [
                # (1) Insert: Insert a space into the word
                WordSwapRandomCharacterInsertion(  # type: ignore
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word
                WordSwapRandomCharacterDeletion(  # type: ignore
                    random_one=True,
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (3) Swap: Swap random two adjacent letters in the word
                WordSwapNeighboringCharacterSwap(  # type: ignore
                    random_one=True,
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (4) Substitute-C: Replace characters with visually similar characters
                WordSwapHomoglyphSwap(),  # type: ignore
                # (5) Substitute-W: Replace a word with its topk nearest neighbors
                WordSwapEmbedding(max_candidates=self._top_k),  # type: ignore
            ]
        )

        # Define the constraints for the transformations to ensure valid and meaningful changes
        constraints = [
            RepeatModification(),  # type: ignore
            StopwordModification(),  # type: ignore
        ]
        if self._use_universal_sentence_encoder is True:
            # Universal Sentence Encoder model is used, tensorflow_hub is required
            constraints.append(UniversalSentenceEncoder(threshold=self._semantic_threshold))  # type: ignore

        # Create the augmenter that orchestrates all transformations
        return Augmenter(  # type: ignore
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=self._word_swap_ratio,
            transformations_per_example=self._max_transformations,
        )

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt using TextBugger transformations.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing the transformed text.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError(f"Input type {input_type} not supported. Only 'text' is supported.")

        try:
            # Apply TextBugger transformations using TextAttack augmenter
            # This will generate multiple variants of the input prompt
            transformed_texts = self._augmenter.augment(prompt)

            logger.debug(f"Generated {len(transformed_texts)} transformed texts for prompt: {prompt}")
            logger.debug(f"Transformed texts: {transformed_texts}")

            # TODO: should I return all transformed texts or just one?
            selected_text = str(random.choice(transformed_texts))
            return ConverterResult(output_text=selected_text, output_type="text")

        except Exception as e:
            print(f"Error during TextBugger transformation: {str(e)}")
            # Return original prompt if transformation fails
            return ConverterResult(output_text=prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
