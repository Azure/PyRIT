# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Optional modality verification system for prompt targets.

This module provides runtime capability discovery to determine what modalities
a specific target actually supports, beyond what the API declares as possible.

Usage:
    from pyrit.prompt_target.modality_verification import verify_target_modalities
    
    # Get static API capabilities
    api_capabilities = target.SUPPORTED_INPUT_MODALITIES
    
    # Optionally verify actual model capabilities  
    actual_capabilities = await verify_target_modalities(target)
"""

import logging
from typing import Optional, set as Set
import asyncio

from pyrit.models import PromptDataType, Message, MessagePiece
from pyrit.prompt_target.common.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


async def verify_target_modalities(
    target: PromptTarget,
    test_modalities: Optional[Set[frozenset[PromptDataType]]] = None
) -> Set[frozenset[PromptDataType]]:
    """
    Verify which modality combinations a target actually supports.
    
    This function tests the target with minimal requests to determine actual
    capabilities, trimming down from the static API declarations.
    
    Args:
        target: The prompt target to test
        test_modalities: Specific modalities to test (defaults to target's declared capabilities)
        
    Returns:
        Set of actually supported modality combinations
        
    Example:
        # Test if a GPT model actually supports vision
        actual = await verify_target_capabilities(openai_target)
        # Returns: {frozenset(["text"])} or {frozenset(["text"]), frozenset(["text", "image_path"])}
    """
    if test_modalities is None:
        test_modalities = target.SUPPORTED_INPUT_MODALITIES
    
    verified_capabilities: Set[frozenset[PromptDataType]] = set()
    
    for modality_combination in test_modalities:
        try:
            is_supported = await _test_modality_combination(target, modality_combination)
            if is_supported:
                verified_capabilities.add(modality_combination)
        except Exception as e:
            logger.debug(f"Failed to verify {modality_combination}: {e}")
            # If verification fails, assume not supported
            
    return verified_capabilities


async def _test_modality_combination(
    target: Any, 
    modalities: frozenset[PromptDataType]
) -> bool:
    """
    Test a specific modality combination with minimal API request.
    
    Args:
        target: The target to test
        modalities: The combination of modalities to test
        
    Returns:
        True if the combination is supported, False otherwise
    """
    try:
        # Create a minimal test message for this modality combination
        test_message = _create_test_message(modalities)
        
        # Attempt to send the test message
        await target.send_prompt_async(message=test_message)
        
        return True
        
    except Exception as e:
        # Common error patterns that indicate unsupported modality
        error_msg = str(e).lower()
        unsupported_patterns = [
            "unsupported",
            "invalid",
            "not supported", 
            "cannot process",
            "modality not available"
        ]
        
        if any(pattern in error_msg for pattern in unsupported_patterns):
            logger.info(f"Modality {modalities} not supported: {e}")
            return False
        
        # Other errors might be temporary, so we're conservative and assume supported
        logger.info(f"Unclear error testing {modalities}: {e}")
        return True


def _create_test_message(modalities: frozenset[PromptDataType]) -> Message:
    """
    Create a minimal test message for the specified modalities.
    
    Args:
        modalities: The modalities to include in the test message
        
    Returns:
        A minimal Message object for testing
    """
    pieces = []
    
    if "text" in modalities:
        pieces.append(MessagePiece(data_type="text", data="test"))
    
    if "image_path" in modalities:
        # Use a minimal test image data URL (1x1 transparent pixel)
        test_image_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        pieces.append(MessagePiece(data_type="image_path", data=test_image_data))
    
    if "audio_path" in modalities:
        # Use minimal test audio data if needed
        pieces.append(MessagePiece(data_type="audio_path", data="test_audio_data"))
        
    if "video_path" in modalities:
        # Use minimal test video data if needed  
        pieces.append(MessagePiece(data_type="video_path", data="test_video_data"))
    
    return Message(conversation_id="verification_test", pieces=pieces)