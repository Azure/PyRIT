# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for modality support detection using set[frozenset[PromptDataType]] architecture.

This test suite demonstrates Roman's requested architecture where:
- SUPPORTED_INPUT_MODALITIES is set[frozenset[PromptDataType]]
- Each frozenset represents a valid combination of modalities
- Exact frozenset matching for precise capability detection
"""

import pytest
from unittest.mock import AsyncMock, Mock

from pyrit.models import PromptDataType
from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget
from pyrit.prompt_target.hugging_face.hugging_face_chat_target import HuggingFaceChatTarget
from pyrit.prompt_target.text_target import TextTarget


class TestModalitySupport:
    """Test modality support detection with set[frozenset[PromptDataType]] architecture."""

    def test_text_target_modalities(self):
        """Test TextTarget only supports text."""
        target = TextTarget()
        
        # Text-only should be supported
        assert target.input_modality_supported({"text"})
        
        # Multimodal should not be supported
        assert not target.input_modality_supported({"text", "image_path"})
        assert not target.input_modality_supported({"image_path"})
        assert not target.input_modality_supported({"text", "audio_path"})

    def test_huggingface_target_modalities(self):
        """Test HuggingFace target only supports text."""
        # Mock the necessary components to avoid actual model loading
        with pytest.mock.patch("pyrit.prompt_target.hugging_face.hugging_face_chat_target.AutoTokenizer"):
            with pytest.mock.patch("pyrit.prompt_target.hugging_face.hugging_face_chat_target.AutoModelForCausalLM"):
                target = HuggingFaceChatTarget(model_id="test-model")
                
                # Text-only should be supported
                assert target.input_modality_supported({"text"})
                
                # Multimodal should not be supported  
                assert not target.input_modality_supported({"text", "image_path"})
                assert not target.input_modality_supported({"image_path"})

    def test_openai_vision_model_modalities(self):
        """Test OpenAI vision models support text + image combinations."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        
        # Test GPT-4o model (vision-capable)
        target = OpenAIChatTarget(model_name="gpt-4o")
        target._client = mock_client
        target._async_client = mock_client
        
        # Should support text-only
        assert target.input_modality_supported({"text"})
        
        # Should support text + image
        assert target.input_modality_supported({"text", "image_path"})
        
        # Should NOT support image-only or other combinations
        assert not target.input_modality_supported({"image_path"})
        assert not target.input_modality_supported({"text", "audio_path"})
        assert not target.input_modality_supported({"text", "image_path", "audio_path"})

    def test_openai_text_model_modalities(self):
        """Test OpenAI text-only models."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        
        # Test GPT-3.5 model (text-only)
        target = OpenAIChatTarget(model_name="gpt-3.5-turbo")
        target._client = mock_client
        target._async_client = mock_client
        
        # Should support text-only
        assert target.input_modality_supported({"text"})
        
        # Should NOT support multimodal
        assert not target.input_modality_supported({"text", "image_path"})
        assert not target.input_modality_supported({"image_path"})

    def test_future_proof_model_detection(self):
        """Test future-proof pattern matching for new models."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        
        # Test future model names that should be detected as vision-capable
        future_models = [
            "gpt-5-vision",
            "gpt-4.5-multimodal", 
            "omni-model-v2",
            "custom-vision-model"
        ]
        
        for model_name in future_models:
            target = OpenAIChatTarget(model_name=model_name)
            target._client = mock_client  
            target._async_client = mock_client
            
            # Should detect as multimodal based on keywords
            assert target.input_modality_supported({"text", "image_path"}), f"Model {model_name} should support vision"
            assert target.input_modality_supported({"text"})

    def test_frozenset_exact_matching(self):
        """Test that modality checking uses exact frozenset matching."""
        mock_client = AsyncMock()
        target = OpenAIChatTarget(model_name="gpt-4o")
        target._client = mock_client
        target._async_client = mock_client
        
        # Get the supported modalities
        supported = target.SUPPORTED_INPUT_MODALITIES
        
        # Should contain exactly these frozensets
        expected_modalities = {
            frozenset(["text"]),
            frozenset(["text", "image_path"])
        }
        assert supported == expected_modalities
        
        # Order shouldn't matter in the frozenset
        assert target.input_modality_supported({"image_path", "text"})
        assert target.input_modality_supported({"text", "image_path"})

    def test_output_modality_support(self):
        """Test output modality support using SUPPORTED_OUTPUT_MODALITIES variable."""
        target = TextTarget()
        
        # Should support text output
        assert target.output_modality_supported({"text"})
        
        # Should not support other output types
        assert not target.output_modality_supported({"image_path"})
        assert not target.output_modality_supported({"text", "image_path"})
        
        # Test that it uses the SUPPORTED_OUTPUT_MODALITIES variable
        expected_output = {frozenset(["text"])}
        assert target.SUPPORTED_OUTPUT_MODALITIES == expected_output

    def test_modality_type_validation(self):
        """Test that modality checking works with PromptDataType literals."""
        target = TextTarget()
        
        # Test with actual PromptDataType values
        text_type: PromptDataType = "text"
        image_type: PromptDataType = "image_path"
        audio_type: PromptDataType = "audio_path"
        
        assert target.input_modality_supported({text_type})
        assert not target.input_modality_supported({text_type, image_type})
        assert not target.input_modality_supported({audio_type})