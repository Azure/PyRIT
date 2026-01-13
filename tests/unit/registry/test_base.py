# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pyrit.registry.base import RegistryItemMetadata, _matches_filters


@dataclass(frozen=True)
class MetadataWithTags(RegistryItemMetadata):
    """Test metadata with a tags field for list filtering tests."""

    tags: tuple[str, ...]


class TestMatchesFilters:
    """Tests for the _matches_filters function."""

    def test_matches_filters_exact_match_string(self):
        """Test that exact string matches work."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, include_filters={"name": "test_item"}) is True
        assert _matches_filters(metadata, include_filters={"class_name": "TestClass"}) is True

    def test_matches_filters_no_match_string(self):
        """Test that non-matching strings return False."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, include_filters={"name": "other_item"}) is False
        assert _matches_filters(metadata, include_filters={"class_name": "OtherClass"}) is False

    def test_matches_filters_multiple_filters_all_match(self):
        """Test that all filters must match."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, include_filters={"name": "test_item", "class_name": "TestClass"}) is True

    def test_matches_filters_multiple_filters_partial_match(self):
        """Test that partial matches return False when not all filters match."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, include_filters={"name": "test_item", "class_name": "OtherClass"}) is False

    def test_matches_filters_key_not_in_metadata(self):
        """Test that filtering on a non-existent key returns False."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, include_filters={"nonexistent_key": "value"}) is False

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
        metadata = MetadataWithTags(
            name="test_item",
            class_name="TestClass",
            description="A test item",
            tags=("tag1", "tag2", "tag3"),
        )
        assert _matches_filters(metadata, include_filters={"tags": "tag1"}) is True
        assert _matches_filters(metadata, include_filters={"tags": "tag2"}) is True

    def test_matches_filters_list_value_not_contains_filter(self):
        """Test filtering when metadata value is a list and filter value is not in the list."""
        metadata = MetadataWithTags(
            name="test_item",
            class_name="TestClass",
            description="A test item",
            tags=("tag1", "tag2", "tag3"),
        )
        assert _matches_filters(metadata, include_filters={"tags": "missing_tag"}) is False

    def test_matches_filters_exclude_exact_match(self):
        """Test that exclude filters work for exact matches."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        assert _matches_filters(metadata, exclude_filters={"name": "test_item"}) is False
        assert _matches_filters(metadata, exclude_filters={"name": "other_item"}) is True

    def test_matches_filters_exclude_list_value(self):
        """Test exclude filters work for list values."""
        metadata = MetadataWithTags(
            name="test_item",
            class_name="TestClass",
            description="A test item",
            tags=("tag1", "tag2", "tag3"),
        )
        assert _matches_filters(metadata, exclude_filters={"tags": "tag1"}) is False
        assert _matches_filters(metadata, exclude_filters={"tags": "missing_tag"}) is True

    def test_matches_filters_exclude_nonexistent_key(self):
        """Test that exclude filters for non-existent keys don't exclude the item."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        # Non-existent key in exclude filter should not exclude the item
        assert _matches_filters(metadata, exclude_filters={"nonexistent_key": "value"}) is True

    def test_matches_filters_combined_include_and_exclude(self):
        """Test combined include and exclude filters."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="A test item",
        )
        # Include matches, exclude doesn't -> should pass
        assert (
            _matches_filters(
                metadata, include_filters={"name": "test_item"}, exclude_filters={"class_name": "OtherClass"}
            )
            is True
        )
        # Include matches, exclude also matches -> should fail
        assert (
            _matches_filters(
                metadata, include_filters={"name": "test_item"}, exclude_filters={"class_name": "TestClass"}
            )
            is False
        )
        # Include doesn't match, exclude doesn't match -> should fail (include takes precedence)
        assert (
            _matches_filters(
                metadata, include_filters={"name": "other_item"}, exclude_filters={"class_name": "OtherClass"}
            )
            is False
        )


class TestRegistryItemMetadata:
    """Tests for the RegistryItemMetadata dataclass."""

    def test_registry_item_metadata_creation(self):
        """Test creating a RegistryItemMetadata instance."""
        metadata = RegistryItemMetadata(
            name="test_scorer",
            class_name="TestScorer",
            description="A test scorer for testing",
        )
        assert metadata.name == "test_scorer"
        assert metadata.class_name == "TestScorer"
        assert metadata.description == "A test scorer for testing"

    def test_registry_item_metadata_is_frozen(self):
        """Test that RegistryItemMetadata is immutable."""
        metadata = RegistryItemMetadata(
            name="test_item",
            class_name="TestClass",
            description="Description here",
        )

        with pytest.raises(AttributeError):
            metadata.name = "new_name"  # type: ignore[misc]
