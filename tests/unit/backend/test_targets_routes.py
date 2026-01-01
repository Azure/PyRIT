# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.backend.routes.targets import get_target_types


class TestTargetsRoutes:
    """Test cases for targets routes."""

    @pytest.mark.asyncio
    async def test_get_target_types_returns_all_types(self):
        """Test that get_target_types returns all configured target types."""
        result = await get_target_types()

        # Should have multiple target types
        assert len(result) > 0

        # Check structure of each target type
        for target_type in result:
            assert "id" in target_type
            assert "name" in target_type
            assert "description" in target_type
            assert "default_env_vars" in target_type

            # Each should have endpoint, api_key, and model in default_env_vars
            env_vars = target_type["default_env_vars"]
            assert "endpoint" in env_vars
            assert "api_key" in env_vars
            assert "model" in env_vars

    @pytest.mark.asyncio
    async def test_get_target_types_includes_expected_types(self):
        """Test that expected target types are included."""
        result = await get_target_types()
        target_ids = [t["id"] for t in result]

        # Check for expected target types
        assert "OpenAIChatTarget" in target_ids
        assert "OpenAIImageTarget" in target_ids
        assert "OpenAITTSTarget" in target_ids
        assert "OpenAIVideoTarget" in target_ids
        assert "RealtimeTarget" in target_ids
        assert "OpenAIResponseTarget" in target_ids

    @pytest.mark.asyncio
    async def test_get_target_types_env_vars_format(self):
        """Test that default env vars follow expected naming convention."""
        result = await get_target_types()

        for target_type in result:
            env_vars = target_type["default_env_vars"]

            # Endpoint should end with _ENDPOINT
            assert env_vars["endpoint"].endswith("_ENDPOINT")

            # API key should contain KEY
            assert "KEY" in env_vars["api_key"] or "key" in env_vars["api_key"]

            # Model should end with _MODEL
            assert env_vars["model"].endswith("_MODEL")
