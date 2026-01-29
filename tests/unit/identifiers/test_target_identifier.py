# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for TargetIdentifier-specific functionality.

Note: Base Identifier functionality (hash computation, to_dict/from_dict basics,
frozen/hashable properties) is tested via ScorerIdentifier in test_scorer_identifier.py.
These tests focus on target-specific fields and behaviors.
"""

import pytest

from pyrit.identifiers import TargetIdentifier


class TestTargetIdentifierBasic:
    """Test basic TargetIdentifier functionality."""

    def test_target_identifier_creation_minimal(self):
        """Test creating a TargetIdentifier with only required fields."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
        )

        assert identifier.class_name == "TestTarget"
        assert identifier.class_module == "pyrit.prompt_target.test_target"
        assert identifier.endpoint == ""
        assert identifier.model_name == ""
        assert identifier.temperature is None
        assert identifier.top_p is None
        assert identifier.max_requests_per_minute is None
        assert identifier.target_specific_params is None
        assert identifier.hash is not None
        assert len(identifier.hash) == 64  # SHA256 hex digest length

    def test_target_identifier_unique_name_minimal(self):
        """Test that unique_name is auto-computed with minimal fields (no model_name or endpoint)."""
        identifier = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
        )

        # unique_name format with no model/endpoint: {snake_class_name}::{hash[:8]}
        assert identifier.unique_name == f"open_ai_chat_target::{identifier.hash[:8]}"

    def test_target_identifier_unique_name_with_model(self):
        """Test that unique_name includes model_name when provided."""
        identifier = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
            model_name="gpt-4o",
        )

        # unique_name format: {snake_class_name}::{model_name}::{hash[:8]}
        assert identifier.unique_name == f"open_ai_chat_target::gpt-4o::{identifier.hash[:8]}"

    def test_target_identifier_unique_name_with_endpoint(self):
        """Test that unique_name includes endpoint host when provided."""
        identifier = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
            endpoint="https://api.openai.com/v1/chat/completions",
        )

        # unique_name format: {snake_class_name}::{endpoint_host}::{hash[:8]}
        assert identifier.unique_name == f"open_ai_chat_target::api.openai.com::{identifier.hash[:8]}"

    def test_target_identifier_unique_name_with_model_and_endpoint(self):
        """Test that unique_name includes both model_name and endpoint when provided."""
        identifier = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
            model_name="gpt-4o",
            endpoint="https://api.openai.com/v1/chat/completions",
        )

        # unique_name format: {snake_class_name}::{model_name}::{endpoint_host}::{hash[:8]}
        assert identifier.unique_name == f"open_ai_chat_target::gpt-4o::api.openai.com::{identifier.hash[:8]}"

    def test_target_identifier_creation_all_fields(self):
        """Test creating a TargetIdentifier with all fields."""
        identifier = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
            endpoint="https://api.openai.com/v1",
            model_name="gpt-4o",
            temperature=0.7,
            top_p=0.9,
            max_requests_per_minute=100,
            target_specific_params={"max_tokens": 1000, "headers": {}},
        )

        assert identifier.endpoint == "https://api.openai.com/v1"
        assert identifier.model_name == "gpt-4o"
        assert identifier.temperature == 0.7
        assert identifier.top_p == 0.9
        assert identifier.max_requests_per_minute == 100
        assert identifier.target_specific_params["max_tokens"] == 1000


class TestTargetIdentifierSpecificFields:
    """Test TargetIdentifier-specific fields: endpoint, model_name, temperature, top_p, target_specific_params."""

    def test_endpoint_stored_correctly(self):
        """Test that endpoint is stored correctly."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            endpoint="https://example.com/api",
        )

        assert identifier.endpoint == "https://example.com/api"

    def test_model_name_stored_correctly(self):
        """Test that model_name is stored correctly."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            model_name="gpt-4o-mini",
        )

        assert identifier.model_name == "gpt-4o-mini"

    def test_temperature_and_top_p_stored_correctly(self):
        """Test that temperature and top_p are stored correctly."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            temperature=0.5,
            top_p=0.95,
        )

        assert identifier.temperature == 0.5
        assert identifier.top_p == 0.95

    def test_target_specific_params_stored_correctly(self):
        """Test that target_specific_params are stored correctly."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            target_specific_params={
                "max_tokens": 2000,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            },
        )

        assert identifier.target_specific_params["max_tokens"] == 2000
        assert identifier.target_specific_params["frequency_penalty"] == 0.5
        assert identifier.target_specific_params["presence_penalty"] == 0.3

    def test_max_requests_per_minute_stored_correctly(self):
        """Test that max_requests_per_minute is stored correctly."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            max_requests_per_minute=60,
        )

        assert identifier.max_requests_per_minute == 60


class TestTargetIdentifierHash:
    """Test hash computation for TargetIdentifier."""

    def test_hash_deterministic(self):
        """Test that hash is the same for identical configurations."""
        identifier1 = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            endpoint="https://api.example.com",
            model_name="test-model",
        )
        identifier2 = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            endpoint="https://api.example.com",
            model_name="test-model",
        )

        assert identifier1.hash == identifier2.hash
        assert len(identifier1.hash) == 64  # SHA256 hex digest length

    def test_hash_different_for_different_endpoints(self):
        """Test that different endpoints produce different hashes."""
        base_args = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier1 = TargetIdentifier(endpoint="https://api1.example.com", **base_args)
        identifier2 = TargetIdentifier(endpoint="https://api2.example.com", **base_args)

        assert identifier1.hash != identifier2.hash

    def test_hash_different_for_different_model_names(self):
        """Test that different model names produce different hashes."""
        base_args = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier1 = TargetIdentifier(model_name="gpt-4o", **base_args)
        identifier2 = TargetIdentifier(model_name="gpt-4o-mini", **base_args)

        assert identifier1.hash != identifier2.hash

    def test_hash_different_for_different_temperature(self):
        """Test that different temperature values produce different hashes."""
        base_args = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier1 = TargetIdentifier(temperature=0.7, **base_args)
        identifier2 = TargetIdentifier(temperature=0.9, **base_args)

        assert identifier1.hash != identifier2.hash

    def test_hash_different_for_different_top_p(self):
        """Test that different top_p values produce different hashes."""
        base_args = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier1 = TargetIdentifier(top_p=0.9, **base_args)
        identifier2 = TargetIdentifier(top_p=0.95, **base_args)

        assert identifier1.hash != identifier2.hash

    def test_hash_different_for_different_target_specific_params(self):
        """Test that different target_specific_params produce different hashes."""
        base_args = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier1 = TargetIdentifier(target_specific_params={"max_tokens": 100}, **base_args)
        identifier2 = TargetIdentifier(target_specific_params={"max_tokens": 200}, **base_args)

        assert identifier1.hash != identifier2.hash


class TestTargetIdentifierToDict:
    """Test to_dict method for TargetIdentifier."""

    def test_to_dict_basic(self):
        """Test basic to_dict output."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
        )

        result = identifier.to_dict()

        assert result["class_name"] == "TestTarget"
        assert result["class_module"] == "pyrit.prompt_target.test_target"
        assert result["hash"] == identifier.hash
        # class_description, identifier_type, and unique_name are excluded from storage
        assert "class_description" not in result
        assert "identifier_type" not in result
        assert "unique_name" not in result

    def test_to_dict_includes_endpoint_and_model_name(self):
        """Test that endpoint and model_name are included in to_dict."""
        identifier = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
            endpoint="https://api.openai.com/v1",
            model_name="gpt-4o",
        )

        result = identifier.to_dict()

        assert result["endpoint"] == "https://api.openai.com/v1"
        assert result["model_name"] == "gpt-4o"

    def test_to_dict_includes_temperature_and_top_p_when_set(self):
        """Test that temperature and top_p are included when set."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            temperature=0.7,
            top_p=0.9,
        )

        result = identifier.to_dict()

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_to_dict_excludes_none_values(self):
        """Test that None values are excluded from to_dict."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            # temperature, top_p, max_requests_per_minute, target_specific_params are None
        )

        result = identifier.to_dict()

        assert "temperature" not in result
        assert "top_p" not in result
        assert "max_requests_per_minute" not in result
        assert "target_specific_params" not in result

    def test_to_dict_includes_target_specific_params(self):
        """Test that target_specific_params are included in to_dict."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            target_specific_params={"max_tokens": 1000, "seed": 42},
        )

        result = identifier.to_dict()

        assert result["target_specific_params"] == {"max_tokens": 1000, "seed": 42}


class TestTargetIdentifierFromDict:
    """Test from_dict method for TargetIdentifier."""

    def test_from_dict_basic(self):
        """Test creating TargetIdentifier from a basic dict."""
        data = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier = TargetIdentifier.from_dict(data)

        assert identifier.class_name == "TestTarget"
        # unique_name is auto-computed
        assert identifier.unique_name.startswith("test_target::")

    def test_from_dict_with_all_target_fields(self):
        """Test creating TargetIdentifier from dict with all target-specific fields."""
        data = {
            "class_name": "OpenAIChatTarget",
            "class_module": "pyrit.prompt_target.openai.openai_chat_target",
            "class_description": "OpenAI chat target",
            "identifier_type": "instance",
            "endpoint": "https://api.openai.com/v1",
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_requests_per_minute": 60,
            "target_specific_params": {"max_tokens": 1000},
        }

        identifier = TargetIdentifier.from_dict(data)

        assert identifier.endpoint == "https://api.openai.com/v1"
        assert identifier.model_name == "gpt-4o"
        assert identifier.temperature == 0.7
        assert identifier.top_p == 0.9
        assert identifier.max_requests_per_minute == 60
        assert identifier.target_specific_params["max_tokens"] == 1000

    def test_from_dict_handles_legacy_type_key(self):
        """Test that from_dict handles legacy '__type__' key."""
        data = {
            "__type__": "TestTarget",  # Legacy key
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        identifier = TargetIdentifier.from_dict(data)

        assert identifier.class_name == "TestTarget"

    def test_from_dict_handles_deprecated_type_key(self):
        """Test that from_dict handles deprecated 'type' key with warning."""
        data = {
            "type": "TestTarget",  # Deprecated key
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
        }

        with pytest.warns(DeprecationWarning, match="'type' key in Identifier dict is deprecated"):
            identifier = TargetIdentifier.from_dict(data)

        assert identifier.class_name == "TestTarget"

    def test_from_dict_ignores_unknown_fields(self):
        """Test that from_dict ignores fields not in the dataclass."""
        data = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
            "unknown_field": "should be ignored",
            "hash": "abc123stored_hash_preserved",
            "unique_name": "stored_name_ignored_because_recomputed",
        }

        identifier = TargetIdentifier.from_dict(data)

        assert identifier.class_name == "TestTarget"
        # hash is preserved from dict (not recomputed) to handle truncated fields
        assert identifier.hash == "abc123stored_hash_preserved"
        # unique_name is recomputed from hash
        assert identifier.unique_name == f"test_target::{identifier.hash[:8]}"

    def test_from_dict_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip works."""
        original = TargetIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            class_description="OpenAI chat target",
            identifier_type="instance",
            endpoint="https://api.openai.com/v1",
            model_name="gpt-4o",
            temperature=0.7,
            target_specific_params={"max_tokens": 500},
        )

        storage_dict = original.to_dict()
        # Add back the excluded fields for reconstruction
        storage_dict["class_description"] = "OpenAI chat target"
        storage_dict["identifier_type"] = "instance"

        reconstructed = TargetIdentifier.from_dict(storage_dict)

        assert reconstructed.class_name == original.class_name
        assert reconstructed.endpoint == original.endpoint
        assert reconstructed.model_name == original.model_name
        assert reconstructed.temperature == original.temperature
        assert reconstructed.target_specific_params == original.target_specific_params
        # Hash should match since config is the same
        assert reconstructed.hash == original.hash

    def test_from_dict_provides_defaults_for_missing_fields(self):
        """Test that from_dict provides defaults for missing optional fields."""
        data = {
            "class_name": "LegacyTarget",
            "class_module": "pyrit.prompt_target.legacy",
            "class_description": "A legacy target",
            "identifier_type": "instance",
            # Missing endpoint, model_name, temperature, top_p, etc.
        }

        identifier = TargetIdentifier.from_dict(data)

        assert identifier.endpoint == ""
        assert identifier.model_name == ""
        assert identifier.temperature is None
        assert identifier.top_p is None
        assert identifier.max_requests_per_minute is None
        assert identifier.target_specific_params is None


class TestTargetIdentifierFrozen:
    """Test that TargetIdentifier is immutable (frozen)."""

    def test_cannot_modify_fields(self):
        """Test that attempting to modify fields raises an error."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            endpoint="https://api.example.com",
        )

        with pytest.raises(AttributeError):
            identifier.class_name = "ModifiedTarget"

        with pytest.raises(AttributeError):
            identifier.endpoint = "https://modified.example.com"

        with pytest.raises(AttributeError):
            identifier.temperature = 0.5

    def test_can_use_as_dict_key(self):
        """Test that frozen identifier can be used as dict key (hashable)."""
        identifier = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
        )

        # Should be hashable and usable as dict key
        d = {identifier: "value"}
        assert d[identifier] == "value"


class TestTargetIdentifierNormalize:
    """Test the normalize class method for TargetIdentifier."""

    def test_normalize_returns_target_identifier_unchanged(self):
        """Test that normalize returns a TargetIdentifier unchanged."""
        original = TargetIdentifier(
            class_name="TestTarget",
            class_module="pyrit.prompt_target.test_target",
            class_description="A test target",
            identifier_type="instance",
            endpoint="https://api.example.com",
        )

        result = TargetIdentifier.normalize(original)

        assert result is original
        assert result.endpoint == "https://api.example.com"

    def test_normalize_converts_dict_to_target_identifier(self):
        """Test that normalize converts a dict to TargetIdentifier with deprecation warning."""
        data = {
            "class_name": "TestTarget",
            "class_module": "pyrit.prompt_target.test_target",
            "class_description": "A test target",
            "identifier_type": "instance",
            "endpoint": "https://api.example.com",
            "model_name": "gpt-4o",
        }

        with pytest.warns(DeprecationWarning, match="dict for TargetIdentifier is deprecated"):
            result = TargetIdentifier.normalize(data)

        assert isinstance(result, TargetIdentifier)
        assert result.class_name == "TestTarget"
        assert result.endpoint == "https://api.example.com"
        assert result.model_name == "gpt-4o"

    def test_normalize_raises_for_invalid_type(self):
        """Test that normalize raises TypeError for invalid input types."""
        with pytest.raises(TypeError, match="Expected TargetIdentifier or dict"):
            TargetIdentifier.normalize("invalid")

        with pytest.raises(TypeError, match="Expected TargetIdentifier or dict"):
            TargetIdentifier.normalize(123)

        with pytest.raises(TypeError, match="Expected TargetIdentifier or dict"):
            TargetIdentifier.normalize(["list", "of", "values"])
