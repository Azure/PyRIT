#!/usr/bin/env python3

# Simple test script for modality support

from pyrit.prompt_target.text_target import TextTarget
from pyrit.memory import CentralMemory, SQLiteMemory
from unittest.mock import AsyncMock
import tempfile
import os

# Set up memory
temp_dir = tempfile.mkdtemp()
memory = SQLiteMemory(db_path=":memory:")
CentralMemory.set_memory_instance(memory)

# Test TextTarget
print("=== Testing TextTarget ===")
target = TextTarget()
print(f"Text support: {target.input_modality_supported({'text'})}")
print(f"Multimodal support: {target.input_modality_supported({'text', 'image_path'})}")
print(f"SUPPORTED_INPUT_MODALITIES: {target.SUPPORTED_INPUT_MODALITIES}")

# Test OpenAI targets
print("\n=== Testing OpenAI Targets ===")
try:
    from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget
    
    # Mock the client to avoid actual API calls
    mock_client = AsyncMock()
    
    # Test vision model
    vision_target = OpenAIChatTarget(model_name="gpt-4o")
    vision_target._client = mock_client
    vision_target._async_client = mock_client
    
    print(f"GPT-4o text support: {vision_target.input_modality_supported({'text'})}")
    print(f"GPT-4o vision support: {vision_target.input_modality_supported({'text', 'image_path'})}")
    print(f"GPT-4o SUPPORTED_INPUT_MODALITIES: {vision_target.SUPPORTED_INPUT_MODALITIES}")
    
    # Test text-only model
    text_target = OpenAIChatTarget(model_name="gpt-3.5-turbo")
    text_target._client = mock_client
    text_target._async_client = mock_client
    
    print(f"GPT-3.5 text support: {text_target.input_modality_supported({'text'})}")
    print(f"GPT-3.5 vision support: {text_target.input_modality_supported({'text', 'image_path'})}")
    print(f"GPT-3.5 SUPPORTED_INPUT_MODALITIES: {text_target.SUPPORTED_INPUT_MODALITIES}")
    
except Exception as e:
    print(f"OpenAI test failed: {e}")

print("\n=== Test Complete ===")