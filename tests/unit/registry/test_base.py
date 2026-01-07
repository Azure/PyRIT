# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.registry.base import RegistryItemMetadata, _matches_filters


class TestMatchesFilters:
    """Tests for the _matches_filters function."""

    def test_matches_filters_exact_match_string(self):
        """Test that exact string matches work."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, name="test_item") is True
        assert _matches_filters(metadata, class_name="TestClass") is True

    def test_matches_filters_no_match_string(self):
        """Test that non-matching strings return False."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, name="other_item") is False
        assert _matches_filters(metadata, class_name="OtherClass") is False

    def test_matches_filters_multiple_filters_all_match(self):
        """Test that all filters must match."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, name="test_item", class_name="TestClass") is True

    def test_matches_filters_multiple_filters_partial_match(self):
        """Test that partial matches return False when not all filters match."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, name="test_item", class_name="OtherClass") is False

    def test_matches_filters_key_not_in_metadata(self):
        """Test that filtering on a non-existent key returns False."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, nonexistent_key="value") is False

    def test_matches_filters_empty_filters(self):
        """Test that empty filters return True."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata) is True

    def test_matches_filters_list_value_contains_filter(self):
        """Test filtering when metadata value is a list and filter value is in the list."""
        metadata = {
            "name": "test_item",
            "class_name": "TestClass",
            "description": "A test item",
            "tags": ["tag1", "tag2", "tag3"],
        }
        assert _matches_filters(metadata, tags="tag1") is True
        assert _matches_filters(metadata, tags="tag2") is True

    def test_matches_filters_list_value_not_contains_filter(self):
        """Test filtering when metadata value is a list and filter value is not in the list."""
        metadata = {
            "name": "test_item",
            "class_name": "TestClass",
            "description": "A test item",
            "tags": ["tag1", "tag2", "tag3"],
        }
        assert _matches_filters(metadata, tags="missing_tag") is False


class TestRegistryItemMetadata:
    """Tests for the RegistryItemMetadata TypedDict."""

    def test_registry_item_metadata_creation(self):
        """Test creating a RegistryItemMetadata instance."""
        metadata = RegistryItemMetadata(
            name="test_scorer",
            class_name="TestScorer",
            description="A test scorer for testing",
        )
        assert metadata["name"] == "test_scorer"
        assert metadata["class_name"] == "TestScorer"
        assert metadata["description"] == "A test scorer for testing"

    def test_registry_item_metadata_as_dict(self):
        """Test that RegistryItemMetadata behaves as a dict."""
        metadata: RegistryItemMetadata = {
            "name": "test_item",
            "class_name": "TestClass",
            "description": "Description here",
        }

        assert "name" in metadata
        assert len(metadata) == 3
        assert list(metadata.keys()) == ["name", "class_name", "description"]
