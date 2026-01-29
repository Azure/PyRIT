# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field

import pytest

import pyrit
from pyrit.identifiers import Identifier, LegacyIdentifiable
from pyrit.identifiers.identifier import _EXCLUDE, _ExcludeFrom, _expand_exclusions


class TestLegacyIdentifiable:
    """Tests for the LegacyIdentifiable abstract base class."""

    def test_legacy_identifiable_get_identifier_is_abstract(self):
        """Test that get_identifier is an abstract method that must be implemented."""

        class ConcreteLegacyIdentifiable(LegacyIdentifiable):
            def get_identifier(self) -> dict[str, str]:
                return {"type": "test", "name": "example"}

        obj = ConcreteLegacyIdentifiable()
        result = obj.get_identifier()
        assert result == {"type": "test", "name": "example"}

    def test_legacy_identifiable_str_returns_identifier_dict(self):
        """Test that __str__ returns the get_identifier() result as a string."""

        class ConcreteLegacyIdentifiable(LegacyIdentifiable):
            def get_identifier(self) -> dict[str, str]:
                return {"type": "test"}

        obj = ConcreteLegacyIdentifiable()
        # __str__ returns the identifier dict as a string
        assert str(obj) == "{'type': 'test'}"


class TestIdentifier:
    """Tests for the Identifier dataclass."""

    def test_identifier_creation(self):
        """Test creating an Identifier instance with all required fields."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestScorer",
            class_module="pyrit.test.scorer",
            class_description="A test scorer for testing",
        )
        assert identifier.identifier_type == "class"
        assert identifier.class_name == "TestScorer"
        assert identifier.class_module == "pyrit.test.scorer"
        assert identifier.class_description == "A test scorer for testing"
        # unique_name is auto-computed from class_name and hash
        assert identifier.unique_name is not None
        assert "test_scorer" in identifier.unique_name

    def test_identifier_is_frozen(self):
        """Test that Identifier is immutable."""
        identifier = Identifier(
            identifier_type="instance",
            class_name="TestClass",
            class_module="test.module",
            class_description="Description here",
        )

        with pytest.raises(AttributeError):
            identifier.unique_name = "new_name"  # type: ignore[misc]

    def test_identifier_type_literal_class(self):
        """Test identifier_type with 'class' value."""
        identifier = Identifier(
            identifier_type="class",
            class_name="Test",
            class_module="test",
            class_description="",
        )
        assert identifier.identifier_type == "class"

    def test_identifier_type_literal_instance(self):
        """Test identifier_type with 'instance' value."""
        identifier = Identifier(
            identifier_type="instance",
            class_name="Test",
            class_module="test",
            class_description="",
        )
        assert identifier.identifier_type == "instance"


class TestIdentifierHash:
    """Tests for Identifier hash computation."""

    def test_hash_computed_at_creation(self):
        """Test that hash is computed when the Identifier is created."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
        )
        assert identifier.hash is not None
        assert len(identifier.hash) == 64  # SHA256 hex length

    def test_hash_is_deterministic(self):
        """Test that the same inputs produce the same hash."""
        identifier1 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
        )
        identifier2 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
        )
        assert identifier1.hash == identifier2.hash

    def test_hash_differs_for_different_storable_fields(self):
        """Test that different storable field values produce different hashes."""
        identifier1 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="Description",
        )
        identifier2 = Identifier(
            identifier_type="class",
            class_name="DifferentClass",
            class_module="test.module",
            class_description="Description",
        )
        assert identifier1.hash != identifier2.hash

    def test_hash_excludes_class_description(self):
        """Test that class_description is excluded from hash computation."""
        identifier1 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="First description",
        )
        identifier2 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="Completely different description",
        )
        # Hash should be the same since class_description is excluded
        assert identifier1.hash == identifier2.hash

    def test_hash_excludes_identifier_type(self):
        """Test that identifier_type is excluded from hash computation."""
        identifier1 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="Description",
        )
        identifier2 = Identifier(
            identifier_type="instance",
            class_name="TestClass",
            class_module="test.module",
            class_description="Description",
        )
        # Hash should be the same since identifier_type is excluded
        assert identifier1.hash == identifier2.hash

    def test_hash_is_immutable(self):
        """Test that the hash cannot be modified."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
        )
        with pytest.raises(AttributeError):
            identifier.hash = "new_hash"  # type: ignore[misc]


class TestIdentifierStorage:
    """Tests for Identifier storage functionality."""

    def test_to_dict_excludes_marked_fields(self):
        """Test that to_dict excludes fields marked with _EXCLUDE containing _ExcludeFrom.STORAGE."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
        )
        storage_dict = identifier.to_dict()

        # Should include storable fields
        assert "class_name" in storage_dict
        assert "class_module" in storage_dict
        assert "hash" in storage_dict
        assert "pyrit_version" in storage_dict

        # Should exclude non-storable fields (marked with _ExcludeFrom.STORAGE)
        assert "class_description" not in storage_dict
        assert "identifier_type" not in storage_dict
        assert "snake_class_name" not in storage_dict
        assert "unique_name" not in storage_dict

    def test_to_dict_values_match(self):
        """Test that to_dict values match the original identifier."""
        identifier = Identifier(
            identifier_type="instance",
            class_name="MyScorer",
            class_module="pyrit.score.my_scorer",
            class_description="My custom scorer",
        )
        storage_dict = identifier.to_dict()

        # unique_name and snake_class_name are excluded from storage
        assert "unique_name" not in storage_dict
        assert "snake_class_name" not in storage_dict
        # Stored fields match
        assert storage_dict["class_name"] == "MyScorer"
        assert storage_dict["class_module"] == "pyrit.score.my_scorer"
        assert storage_dict["hash"] == identifier.hash


class TestIdentifierSubclass:
    """Tests for Identifier subclassing behavior."""

    def test_subclass_inherits_hash_computation(self):
        """Test that subclasses of Identifier also get a computed hash."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            extra_field: str = field(kw_only=True)

        extended = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="A test description",
            extra_field="extra_value",
        )
        assert extended.hash is not None
        assert len(extended.hash) == 64

    def test_subclass_extra_fields_included_in_hash(self):
        """Test that subclass extra fields (not marked) are included in hash."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            extra_field: str = field(kw_only=True)

        extended1 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            extra_field="value1",
        )
        extended2 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            extra_field="value2",
        )
        # Different extra_field values should produce different hashes
        assert extended1.hash != extended2.hash

    def test_subclass_excluded_fields_not_in_hash(self):
        """Test that subclass fields with _ExcludeFrom.STORAGE in _EXCLUDE are excluded from hash via expansion."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            # Only need STORAGE - HASH is implied via expansion
            display_only: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.STORAGE}})

        extended1 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            display_only="display1",
        )
        extended2 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            display_only="display2",
        )
        # display_only is excluded, so hashes should match
        assert extended1.hash == extended2.hash

    def test_subclass_to_dict_includes_extra_storable_fields(self):
        """Test that to_dict includes subclass storable fields."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            extra_field: str = field(kw_only=True)
            # Only need STORAGE - HASH is implied via expansion
            display_only: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.STORAGE}})

        extended = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            extra_field="extra_value",
            display_only="display_value",
        )
        storage_dict = extended.to_dict()

        # Extra storable field should be included
        assert "extra_field" in storage_dict
        assert storage_dict["extra_field"] == "extra_value"

        # Display-only field should be excluded
        assert "display_only" not in storage_dict

    def test_to_dict_excludes_none_and_empty_values(self):
        """Test that to_dict excludes None and empty values."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            optional_str: str = ""
            optional_none: str | None = None
            optional_list: list = field(default_factory=list)
            optional_dict: dict = field(default_factory=dict)
            populated_field: str = "has_value"

        extended = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
        )
        storage_dict = extended.to_dict()

        # Empty/None fields should be excluded
        assert "optional_str" not in storage_dict
        assert "optional_none" not in storage_dict
        assert "optional_list" not in storage_dict
        assert "optional_dict" not in storage_dict

        # Populated field should be included
        assert "populated_field" in storage_dict
        assert storage_dict["populated_field"] == "has_value"


class TestIdentifierFromDict:
    """Tests for Identifier.from_dict functionality."""

    def test_from_dict_preserves_hash_when_provided(self):
        """Test that from_dict preserves the hash from the dict rather than recomputing it.

        This is important for roundtripping identifiers with truncated fields
        (via max_storage_length) where the original hash must be preserved.
        """
        stored_hash = "abc123def456789012345678901234567890123456789012345678901234"
        data = {
            "class_name": "TestClass",
            "class_module": "test.module",
            "hash": stored_hash,
        }

        identifier = Identifier.from_dict(data)

        # Hash should be preserved from the dict, not recomputed
        assert identifier.hash == stored_hash
        # unique_name should use the preserved hash
        assert identifier.unique_name == f"test_class::{stored_hash[:8]}"

    def test_from_dict_computes_hash_when_not_provided(self):
        """Test that from_dict computes the hash when not provided in the dict."""
        data = {
            "class_name": "TestClass",
            "class_module": "test.module",
        }

        identifier = Identifier.from_dict(data)

        # Hash should be computed (64 char SHA256 hex)
        assert identifier.hash is not None
        assert len(identifier.hash) == 64
        # unique_name should use the computed hash
        assert identifier.unique_name == f"test_class::{identifier.hash[:8]}"


class TestPyritVersion:
    """Tests for the pyrit_version field on Identifier."""

    def test_pyrit_version_is_set_by_default(self):
        """Test that pyrit_version is automatically set to the current pyrit version."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
        )
        assert identifier.pyrit_version == pyrit.__version__

    def test_pyrit_version_can_be_overridden(self):
        """Test that pyrit_version can be explicitly provided."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
            pyrit_version="1.0.0",
        )
        assert identifier.pyrit_version == "1.0.0"

    def test_pyrit_version_is_excluded_from_hash(self):
        """Test that pyrit_version is excluded from hash computation."""
        identifier1 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="Description",
            pyrit_version="1.0.0",
        )
        identifier2 = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="Description",
            pyrit_version="2.0.0",
        )
        # Hash should be the same since pyrit_version is excluded
        assert identifier1.hash == identifier2.hash

    def test_pyrit_version_is_included_in_storage(self):
        """Test that pyrit_version is included in to_dict output."""
        identifier = Identifier(
            identifier_type="class",
            class_name="TestClass",
            class_module="test.module",
            class_description="A test description",
            pyrit_version="1.0.0",
        )
        storage_dict = identifier.to_dict()
        assert "pyrit_version" in storage_dict
        assert storage_dict["pyrit_version"] == "1.0.0"

    def test_from_dict_preserves_pyrit_version(self):
        """Test that from_dict preserves the pyrit_version from the dict."""
        data = {
            "class_name": "TestClass",
            "class_module": "test.module",
            "pyrit_version": "0.5.0",
        }

        identifier = Identifier.from_dict(data)
        assert identifier.pyrit_version == "0.5.0"

    def test_from_dict_defaults_pyrit_version_when_missing(self):
        """Test that from_dict defaults pyrit_version to current version when not in dict."""
        data = {
            "class_name": "TestClass",
            "class_module": "test.module",
        }

        identifier = Identifier.from_dict(data)
        assert identifier.pyrit_version == pyrit.__version__


class TestExcludeMetadata:
    """Tests for the _EXCLUDE metadata field configuration."""

    def test_storage_exclusion_implies_hash_exclusion(self):
        """Test that STORAGE exclusion automatically implies HASH exclusion via expansion.

        This validates the catalog expansion pattern where _ExcludeFrom.STORAGE.expands_to
        returns {STORAGE, HASH}, ensuring fields excluded from storage are also excluded
        from hash computation.
        """
        # Verify the expansion catalog works correctly
        assert _ExcludeFrom.HASH in _ExcludeFrom.STORAGE.expands_to
        assert _ExcludeFrom.STORAGE in _ExcludeFrom.STORAGE.expands_to

        # Verify HASH only expands to itself
        assert _ExcludeFrom.HASH.expands_to == {_ExcludeFrom.HASH}

        # Verify _expand_exclusions works on sets
        expanded = _expand_exclusions({_ExcludeFrom.STORAGE})
        assert _ExcludeFrom.HASH in expanded
        assert _ExcludeFrom.STORAGE in expanded

    def test_subclass_storage_only_exclusion_works_via_expansion(self):
        """Test that subclass fields with only _ExcludeFrom.STORAGE work correctly via expansion.

        With the catalog expansion pattern, specifying only STORAGE automatically implies HASH.
        This is the recommended pattern - no need to explicitly specify both.
        """

        @dataclass(frozen=True)
        class ValidIdentifier(Identifier):
            # Only STORAGE is needed - HASH is implied via expansion
            transient_field: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.STORAGE}})

        id1 = ValidIdentifier(
            class_name="Test",
            class_module="test",
            identifier_type="class",
            class_description="desc",
            transient_field="value1",
        )
        id2 = ValidIdentifier(
            class_name="Test",
            class_module="test",
            identifier_type="class",
            class_description="desc",
            transient_field="value2",
        )

        # Hash should be the same (HASH exclusion is implied by STORAGE)
        assert id1.hash == id2.hash

        # Field should not be in storage
        assert "transient_field" not in id1.to_dict()

    def test_hash_only_exclusion_works(self):
        """Test that a field can be excluded from hash only (still stored)."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            metadata_field: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.HASH}})

        extended1 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            metadata_field="value1",
        )
        extended2 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            metadata_field="value2",
        )

        # Hash should be the same since metadata_field is excluded from hash
        assert extended1.hash == extended2.hash

        # But both values should be in storage
        storage1 = extended1.to_dict()
        storage2 = extended2.to_dict()
        assert storage1["metadata_field"] == "value1"
        assert storage2["metadata_field"] == "value2"

    def test_storage_exclusion_works(self):
        """Test that a field excluded from storage is also excluded from hash via expansion."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            # Only need STORAGE - HASH is implied via expansion
            transient_field: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.STORAGE}})

        extended1 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            transient_field="value1",
        )
        extended2 = ExtendedIdentifier(
            class_name="TestClass",
            class_module="test.module",
            identifier_type="class",
            class_description="Description",
            transient_field="value2",
        )

        # Hash should be the same since transient_field is excluded from hash
        assert extended1.hash == extended2.hash

        # Field should not be in storage
        storage1 = extended1.to_dict()
        assert "transient_field" not in storage1
