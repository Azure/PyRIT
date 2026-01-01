# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import patch

import pytest

from pyrit.backend.routes.config import get_env_var_value, get_env_vars


class TestConfigRoutes:
    """Test cases for config routes - especially security filtering."""

    @pytest.mark.asyncio
    async def test_get_env_vars_categorizes_correctly(self):
        """Test that env vars are correctly categorized into keys, endpoints, and models."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_CHAT_KEY": "secret-key-123",
                "AZURE_SECRET": "another-secret",
                "OPENAI_CHAT_ENDPOINT": "https://api.openai.com/v1",
                "AZURE_ENDPOINT": "https://test.azure.com",
                "OPENAI_CHAT_MODEL": "gpt-4",
                "AZURE_MODEL": "gpt-4-turbo",
                "SOME_OTHER_VAR": "not-sensitive",
            },
            clear=True,
        ):
            result = await get_env_vars()

            # Check keys category - should only have names, not values
            assert "OPENAI_CHAT_KEY" in result["keys"]
            assert "AZURE_SECRET" in result["keys"]
            assert len(result["keys"]) == 2

            # Check endpoints category - should not include KEY or SECRET vars
            assert "OPENAI_CHAT_ENDPOINT" in result["endpoints"]
            assert "AZURE_ENDPOINT" in result["endpoints"]
            assert "OPENAI_CHAT_KEY" not in result["endpoints"]
            assert len(result["endpoints"]) == 2

            # Check models category - should not include KEY or SECRET vars
            assert "OPENAI_CHAT_MODEL" in result["models"]
            assert "AZURE_MODEL" in result["models"]
            assert "OPENAI_CHAT_KEY" not in result["models"]
            assert len(result["models"]) == 2

    @pytest.mark.asyncio
    async def test_get_env_var_value_masks_sensitive_vars(self):
        """Test that sensitive env var values are never exposed."""
        with patch.dict(
            os.environ,
            {
                "TEST_API_KEY": "super-secret-key",
                "TEST_SECRET": "another-secret",
                "OPENAI_KEY": "openai-secret",
                "TEST_ENDPOINT": "https://test.com",
            },
            clear=True,
        ):
            # Test API key is masked
            result = await get_env_var_value("TEST_API_KEY")
            assert result["name"] == "TEST_API_KEY"
            assert result["value"] is None
            assert result["masked"] is True
            assert result["exists"] is True

            # Test SECRET is masked
            result = await get_env_var_value("TEST_SECRET")
            assert result["value"] is None
            assert result["masked"] is True

            # Test KEY is masked (case insensitive)
            result = await get_env_var_value("OPENAI_KEY")
            assert result["value"] is None
            assert result["masked"] is True

            # Test non-sensitive var returns value
            result = await get_env_var_value("TEST_ENDPOINT")
            assert result["name"] == "TEST_ENDPOINT"
            assert result["value"] == "https://test.com"
            assert result["masked"] is False
            assert result["exists"] is True

    @pytest.mark.asyncio
    async def test_get_env_var_value_nonexistent_var(self):
        """Test that nonexistent env vars return exists=False."""
        with patch.dict(os.environ, {}, clear=True):
            result = await get_env_var_value("NONEXISTENT_VAR")
            assert result["name"] == "NONEXISTENT_VAR"
            assert result["value"] is None
            assert result["exists"] is False

    @pytest.mark.asyncio
    async def test_key_vars_never_have_values_in_response(self):
        """Critical security test: ensure KEY/SECRET values are NEVER exposed."""
        with patch.dict(
            os.environ,
            {
                "MY_API_KEY": "this-should-never-be-exposed",
                "SECRET_TOKEN": "also-should-not-be-exposed",
                "PRIVATE_KEY": "never-expose-this",
            },
            clear=True,
        ):
            result = await get_env_vars()

            # Keys should be in the keys list
            assert "MY_API_KEY" in result["keys"]
            assert "SECRET_TOKEN" in result["keys"]
            assert "PRIVATE_KEY" in result["keys"]

            # But should NOT appear in endpoints or models (even as names)
            assert "MY_API_KEY" not in result["endpoints"]
            assert "SECRET_TOKEN" not in result["endpoints"]
            assert "PRIVATE_KEY" not in result["models"]

            # And the actual values should NEVER appear anywhere in the response
            result_str = str(result)
            assert "this-should-never-be-exposed" not in result_str
            assert "also-should-not-be-exposed" not in result_str
            assert "never-expose-this" not in result_str
