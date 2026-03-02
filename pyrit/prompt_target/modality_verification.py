# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Optional modality verification system for prompt targets.

This module provides runtime modality discovery to determine what modalities
a specific target actually supports, beyond what the API declares as possible.

Usage:
    from pyrit.prompt_target.modality_verification import verify_target_modalities

    # Get static API modalities
    api_modalities = target.SUPPORTED_INPUT_MODALITIES

    # Optionally verify actual model modalities
    actual_modalities = await verify_target_modalities(target)
"""

import logging
import os
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import Message, MessagePiece, PromptDataType
from pyrit.prompt_target.common.prompt_target import PromptTarget

logger = logging.getLogger(__name__)

# Path to the assets directory containing test files for modality verification
_ASSETS_DIR = DATASETS_PATH / "modality_test_assets"

# Mapping from PromptDataType to test asset filenames
_TEST_ASSETS: dict[str, str] = {
    "image_path": str(_ASSETS_DIR / "test_image.png"),
    "audio_path": str(_ASSETS_DIR / "test_audio.wav"),
    "video_path": str(_ASSETS_DIR / "test_video.mp4"),
}


async def verify_target_modalities(
    target: PromptTarget,
    test_modalities: Optional[set[frozenset[PromptDataType]]] = None,
) -> set[frozenset[PromptDataType]]:
    """
    Verify which modality combinations a target actually supports.

    This function tests the target with minimal requests to determine actual
    modalities, trimming down from the static API declarations.

    Args:
        target: The prompt target to test
        test_modalities: Specific modalities to test (defaults to target's declared modalities)

    Returns:
        Set of actually supported input modality combinations

    Example:
        actual = await verify_target_modalities(openai_target)
        # Returns: {frozenset(["text"])} or {frozenset(["text"]), frozenset(["text", "image_path"])}
    """
    if test_modalities is None:
        test_modalities = target.SUPPORTED_INPUT_MODALITIES

    verified_modalities: set[frozenset[PromptDataType]] = set()

    for modality_combination in test_modalities:
        try:
            is_supported = await _test_modality_combination(target, modality_combination)
            if is_supported:
                verified_modalities.add(modality_combination)
        except Exception as e:
            logger.info(f"Failed to verify {modality_combination}: {e}")

    return verified_modalities


async def _test_modality_combination(
    target: PromptTarget,
    modalities: frozenset[PromptDataType],
) -> bool:
    """
    Test a specific modality combination with a minimal API request.

    Args:
        target: The target to test
        modalities: The combination of modalities to test

    Returns:
        True if the combination is supported, False otherwise
    """
    test_message = _create_test_message(modalities)

    try:
        responses = await target.send_prompt_async(message=test_message)

        # Check if the response itself indicates an error
        for response in responses:
            for piece in response.message_pieces:
                if piece.response_error != "none":
                    logger.info(f"Modality {modalities} returned error response: {piece.converted_value}")
                    return False

        return True

    except Exception as e:
        logger.info(f"Modality {modalities} not supported: {e}")
        return False


def _create_test_message(modalities: frozenset[PromptDataType]) -> Message:
    """
    Create a minimal test message for the specified modalities.

    Args:
        modalities: The modalities to include in the test message

    Returns:
        A Message object with minimal content for each requested modality

    Raises:
        FileNotFoundError: If a required test asset file is missing
        ValueError: If a modality has no configured test asset or no pieces could be created
    """
    pieces: list[MessagePiece] = []
    conversation_id = "modality-verification-test"

    for modality in modalities:
        if modality == "text":
            pieces.append(
                MessagePiece(
                    role="user",
                    original_value="test",
                    original_value_data_type="text",
                    conversation_id=conversation_id,
                )
            )
        elif modality in _TEST_ASSETS:
            asset_path = _TEST_ASSETS[modality]
            if not os.path.isfile(asset_path):
                raise FileNotFoundError(f"Test asset not found for modality '{modality}': {asset_path}")
            pieces.append(
                MessagePiece(
                    role="user",
                    original_value=asset_path,
                    original_value_data_type=modality,
                    conversation_id=conversation_id,
                )
            )
        else:
            raise ValueError(f"No test asset configured for modality: {modality}")

    if not pieces:
        raise ValueError(f"Could not create test message for modalities: {modalities}")

    return Message(pieces)
