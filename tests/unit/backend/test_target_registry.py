# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import os
from unittest.mock import patch

from pyrit.backend.services.target_registry import TargetRegistry
import pyrit.prompt_target as pt


class TestTargetRegistry:
    """Test cases for TargetRegistry"""

    def test_create_target_instance_openai_chat(self, sqlite_instance):
        """Test creating OpenAIChatTarget instance"""
        with patch.dict(os.environ, {
            "TEST_ENDPOINT": "https://api.openai.com/v1",
            "TEST_KEY": "test-key-123",
            "TEST_MODEL": "gpt-4"
        }):
            target = TargetRegistry.create_target_instance(
                target_type="OpenAIChatTarget",
                endpoint_var="TEST_ENDPOINT",
                key_var="TEST_KEY",
                model_var="TEST_MODEL"
            )
            
            assert target is not None
            assert isinstance(target, pt.OpenAIChatTarget)

    def test_create_target_instance_openai_image(self, sqlite_instance):
        """Test creating OpenAIImageTarget instance"""
        with patch.dict(os.environ, {
            "TEST_IMAGE_ENDPOINT": "https://api.openai.com/v1",
            "TEST_IMAGE_KEY": "test-key-123"
        }):
            target = TargetRegistry.create_target_instance(
                target_type="OpenAIImageTarget",
                endpoint_var="TEST_IMAGE_ENDPOINT",
                key_var="TEST_IMAGE_KEY"
            )
            
            assert target is not None
            assert isinstance(target, pt.OpenAIImageTarget)

    def test_create_target_instance_openai_tts(self, sqlite_instance):
        """Test creating OpenAITTSTarget instance"""
        with patch.dict(os.environ, {
            "TEST_TTS_ENDPOINT": "https://api.openai.com/v1",
            "TEST_TTS_KEY": "test-key-123"
        }):
            target = TargetRegistry.create_target_instance(
                target_type="OpenAITTSTarget",
                endpoint_var="TEST_TTS_ENDPOINT",
                key_var="TEST_TTS_KEY"
            )
            
            assert target is not None
            assert isinstance(target, pt.OpenAITTSTarget)

    def test_create_target_instance_openai_video(self, sqlite_instance):
        """Test creating OpenAIVideoTarget instance"""
        with patch.dict(os.environ, {
            "TEST_VIDEO_ENDPOINT": "https://api.openai.com/v1",
            "TEST_VIDEO_KEY": "test-key-123"
        }):
            target = TargetRegistry.create_target_instance(
                target_type="OpenAIVideoTarget",
                endpoint_var="TEST_VIDEO_ENDPOINT",
                key_var="TEST_VIDEO_KEY"
            )
            
            assert target is not None
            assert isinstance(target, pt.OpenAIVideoTarget)

    def test_create_target_instance_realtime(self, sqlite_instance):
        """Test creating RealtimeTarget instance"""
        with patch.dict(os.environ, {
            "TEST_REALTIME_ENDPOINT": "wss://api.openai.com/v1/realtime",
            "TEST_REALTIME_KEY": "test-key-123"
        }):
            target = TargetRegistry.create_target_instance(
                target_type="RealtimeTarget",
                endpoint_var="TEST_REALTIME_ENDPOINT",
                key_var="TEST_REALTIME_KEY"
            )
            
            assert target is not None
            assert isinstance(target, pt.RealtimeTarget)

    def test_create_target_instance_invalid_type(self):
        """Test that invalid target type raises error"""
        with patch.dict(os.environ, {"TEST_ENDPOINT": "https://api.openai.com/v1"}):
            with pytest.raises(ValueError, match="Unknown target type"):
                TargetRegistry.create_target_instance(
                    target_type="NonExistentTarget",
                    endpoint_var="TEST_ENDPOINT"
                )

    def test_create_target_instance_missing_endpoint(self):
        """Test that missing endpoint raises error"""
        with pytest.raises(ValueError, match="Endpoint is required"):
            TargetRegistry.create_target_instance(
                target_type="OpenAIChatTarget",
                endpoint_var="NONEXISTENT_VAR"
            )

    def test_create_target_instance_with_overrides(self, sqlite_instance):
        """Test creating target with direct overrides instead of env vars"""
        target = TargetRegistry.create_target_instance(
            target_type="OpenAIChatTarget",
            endpoint="https://api.openai.com/v1",
            api_key="override-key",
            model_name="gpt-4"
        )
        
        assert target is not None
        assert isinstance(target, pt.OpenAIChatTarget)

    def test_get_available_targets(self):
        """Test discovering available targets from environment"""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_GPT4_ENDPOINT": "https://test.openai.azure.com/v1",
            "AZURE_OPENAI_GPT4_KEY": "test-key",
            "AZURE_OPENAI_GPT4_MODEL": "gpt-4",
            "SOME_OTHER_VAR": "not-an-endpoint"
        }, clear=True):
            targets = TargetRegistry.get_available_targets()
            
            assert len(targets) == 1
            assert targets[0]["endpoint_var"] == "AZURE_OPENAI_GPT4_ENDPOINT"
            assert "Gpt4" in targets[0]["name"] or "GPT4" in targets[0]["name"]

    def test_target_types_in_config_exist(self):
        """Test that all target types listed in config.py actually exist"""
        # Import the target types from config
        from pyrit.backend.routes.config import get_target_types
        import asyncio
        
        target_types = asyncio.run(get_target_types())
        
        for target_type_info in target_types:
            target_type_id = target_type_info["id"]
            # Verify the class exists in pyrit.prompt_target
            assert hasattr(pt, target_type_id), f"Target type {target_type_id} not found in pyrit.prompt_target"
            
            # Verify we can get the class
            target_class = getattr(pt, target_type_id)
            assert target_class is not None
