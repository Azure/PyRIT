# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from pyrit.common.modality_discovery import verify_target_capabilities


class MockOpenAIChatTarget:
    """Mock OpenAI target for testing without requiring environment variables."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @property
    def SUPPORTED_INPUT_MODALITIES(self) -> set[frozenset[str]]:
        """Static pattern matching for known multimodal models with future-proof heuristics."""
        model_lower = self.model_name.lower()
        
        # Text-only patterns (explicit)
        if any(pattern in model_lower for pattern in [
            "gpt-3.5", "davinci", "curie", "babbage", "ada"
        ]):
            return {frozenset({"text"})}
        
        # Multimodal indicators (vision/image support)
        multimodal_indicators = [
            "vision",       # gpt-4-vision-preview, etc.
            "gpt-4o",       # gpt-4o, gpt-4o-mini, gpt-4o-2024-*, etc.  
            "gpt-4-turbo",  # Often has vision
            "gpt-5",        # Future models likely multimodal
            "gpt-4.5",      # Hypothetical intermediate 
            "multimodal",   # Explicit in name
            "omni",         # Omni-modal models
        ]
        
        if any(indicator in model_lower for indicator in multimodal_indicators):
            return {
                frozenset({"text"}),
                frozenset({"text", "image_path"})
            }
        
        # For unknown GPT-4+ models, assume multimodal (safer for newer models)
        if "gpt-4" in model_lower and not any(old_pattern in model_lower for old_pattern in ["gpt-4-0314", "gpt-4-32k"]):
            return {
                frozenset({"text"}),
                frozenset({"text", "image_path"})
            }
            
        # Conservative default for completely unknown models
        return {frozenset({"text"})}
    
    @property 
    def SUPPORTED_OUTPUT_MODALITIES(self) -> set[frozenset[str]]:
        """OpenAI models currently only produce text outputs."""
        return {frozenset({"text"})}
    
    def input_modality_supported(self, modality: frozenset[str]) -> bool:
        """Check if input modality combination is supported."""
        return modality in self.SUPPORTED_INPUT_MODALITIES
    
    def output_modality_supported(self, modality: frozenset[str]) -> bool:
        """Check if output modality combination is supported."""
        return modality in self.SUPPORTED_OUTPUT_MODALITIES


class TestSimpleModalitySupport:
    """Test the simplified modality support system with static declarations."""
    
    def test_gpt4o_static_modalities(self):
        """Test that GPT-4o correctly declares multimodal support."""
        target = MockOpenAIChatTarget("gpt-4o")
        
        input_modalities = target.SUPPORTED_INPUT_MODALITIES
        output_modalities = target.SUPPORTED_OUTPUT_MODALITIES
        
        # Should support text-only and text+image
        assert frozenset({"text"}) in input_modalities
        assert frozenset({"text", "image_path"}) in input_modalities
        assert len(input_modalities) == 2
        
        # Output is text-only
        assert output_modalities == {frozenset({"text"})}
        
    def test_gpt35_static_modalities(self):
        """Test that GPT-3.5 correctly declares text-only support."""
        target = MockOpenAIChatTarget("gpt-3.5-turbo")
        
        input_modalities = target.SUPPORTED_INPUT_MODALITIES
        output_modalities = target.SUPPORTED_OUTPUT_MODALITIES
        
        # Should support only text
        assert input_modalities == {frozenset({"text"})}
        assert output_modalities == {frozenset({"text"})}
        
    def test_gpt4_vision_static_modalities(self):
        """Test that GPT-4 Vision correctly declares multimodal support."""
        target = MockOpenAIChatTarget("gpt-4-vision-preview")
        
        input_modalities = target.SUPPORTED_INPUT_MODALITIES
        
        # Should support text-only and text+image
        assert frozenset({"text"}) in input_modalities
        assert frozenset({"text", "image_path"}) in input_modalities
        
    def test_optional_verification_utility(self):
        """Test the optional verification utility function."""
        # Mock target with required attributes
        mock_target = Mock()
        mock_target._async_client = Mock()
        mock_target.model_name = "gpt-4o"
        
        # Test the verification function
        result = verify_target_capabilities(mock_target, ["image"])
        
        # Should return a dict with the tested modality
        assert isinstance(result, dict)
        assert "image" in result
        assert isinstance(result["image"], bool)
        
    def test_base_class_helper_methods(self):
        """Test that base class helper methods work with static declarations."""
        target = MockOpenAIChatTarget("gpt-4o")
        
        # Test input modality checking
        assert target.input_modality_supported(frozenset({"text"}))
        assert target.input_modality_supported(frozenset({"text", "image_path"}))
        assert not target.input_modality_supported(frozenset({"text", "audio_path"}))
        
        # Test output modality checking
        assert target.output_modality_supported(frozenset({"text"}))
        assert not target.output_modality_supported(frozenset({"audio_path"}))
        
    def test_model_pattern_matching(self):
        """Test that model pattern matching works correctly with future-proof heuristics."""
        test_cases = [
            # Current multimodal models
            ("gpt-4o", True),
            ("gpt-4o-mini", True), 
            ("gpt-4o-2024-08-06", True),
            ("gpt-4-vision-preview", True),
            ("gpt-4-turbo", True),
            ("gpt-4-turbo-2024-04-09", True),
            
            # Future models (should be detected)
            ("gpt-5", True),
            ("gpt-4.5-preview", True),
            ("gpt-4-omni", True),
            ("gpt-4-multimodal", True),
            ("gpt-4-2024-12-01", True),  # Unknown GPT-4 variants assumed multimodal
            
            # Explicit text-only
            ("gpt-3.5-turbo", False),
            ("gpt-3.5-turbo-16k", False),
            ("text-davinci-003", False),
            
            # Old GPT-4 models (known text-only)
            ("gpt-4-0314", False),
            ("gpt-4-32k", False),
            
            # Unknown models (conservative default)
            ("custom-model", False),
            ("claude-3", False),
        ]
        
        for model_name, should_support_image in test_cases:
            target = MockOpenAIChatTarget(model_name)
            input_modalities = target.SUPPORTED_INPUT_MODALITIES
            
            has_image_support = frozenset({"text", "image_path"}) in input_modalities
            assert has_image_support == should_support_image, f"Failed for model: {model_name}"


class TestVerificationUtility:
    """Test the optional verification utility function."""
    
    def test_verify_missing_attributes(self):
        """Test verification with target missing required attributes."""
        mock_target = Mock()
        # Don't add _async_client or model_name
        
        result = verify_target_capabilities(mock_target, ["image"])
        
        # Should return False for all modalities
        assert result == {"image": False}
        
    def test_verify_default_modalities(self):
        """Test verification with default modalities."""
        mock_target = Mock()
        mock_target._async_client = Mock()
        mock_target.model_name = "test-model"
        
        result = verify_target_capabilities(mock_target)
        
        # Should test default modalities (image, audio)
        assert "image" in result
        assert "audio" in result
        assert len(result) == 2