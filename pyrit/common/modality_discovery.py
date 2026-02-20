# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pyrit.models.literals import PromptDataType

logger = logging.getLogger(__name__)


def verify_target_capabilities(
    target: Any,
    test_modalities: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Optional utility to verify actual modality capabilities of a target via runtime testing.
    
    This is a verification tool that can be used to check if declared capabilities
    match actual API behavior. It's completely optional and separate from the main
    modality support system.
    
    Args:
        target: The target to test (must have _async_client and model_name)
        test_modalities: List of modalities to test (defaults to ["image", "audio"])
        
    Returns:
        Dict mapping modality names to actual support status
    """
    if test_modalities is None:
        test_modalities = ["image", "audio"]
        
    if not hasattr(target, '_async_client') or not hasattr(target, 'model_name'):
        logger.warning(f"Target {type(target).__name__} doesn't support capability testing")
        return {modality: False for modality in test_modalities}
    
    capabilities = {}
    
    try:
        for modality in test_modalities:
            capabilities[modality] = _test_single_modality(target, modality)
    except Exception as e:
        logger.warning(f"Capability verification failed for {target.model_name}: {e}")
        return {modality: False for modality in test_modalities}
        
    logger.info(f"Verified capabilities for {target.model_name}: {capabilities}")
    return capabilities


def _test_single_modality(target: Any, modality: str) -> bool:
    """Test a single modality with minimal API request."""
    try:
        if modality == "image":
            # Minimal 1x1 PNG test
            minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            test_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Test image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{minimal_png}"}}
                ]
            }]
        elif modality == "audio":
            # Minimal audio test (placeholder)
            test_messages = [{
                "role": "user", 
                "content": [{"type": "text", "text": "Test audio?"}]
            }]
        else:
            return False
            
        # Try the request
        async def _test():
            try:
                await target._async_client.chat.completions.create(
                    model=target.model_name,
                    messages=test_messages,
                    max_tokens=1
                )
                return True
            except Exception as e:
                error_msg = str(e).lower()
                # Check for modality-specific errors
                if any(indicator in error_msg for indicator in [
                    "does not support", "not supported", "vision is not supported", "invalid content type"
                ]):
                    return False
                return False  # Any error = not supported
                
        # Run the async test
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _test())
                return future.result(timeout=10)
        except RuntimeError:
            return asyncio.run(_test())
            
    except Exception:
        return False