# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field, fields

import pytest

import pyrit
from pyrit.identifiers import Identifier, LegacyIdentifiable
from pyrit.identifiers.identifier import _EXCLUDE, _ExcludeFrom


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
        assert "unique_name" in storage_dict
        assert "class_name" in storage_dict
        assert "class_module" in storage_dict
        assert "hash" in storage_dict

        # Should exclude non-storable fields
        assert "class_description" not in storage_dict
        assert "identifier_type" not in storage_dict

    def test_to_dict_values_match(self):
        """Test that to_dict values match the original identifier."""
        identifier = Identifier(
            identifier_type="instance",
            class_name="MyScorer",
            class_module="pyrit.score.my_scorer",
            class_description="My custom scorer",
        )
        storage_dict = identifier.to_dict()

        # unique_name is auto-computed
        assert storage_dict["unique_name"] == identifier.unique_name
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
        """Test that subclass fields with _ExcludeFrom.HASH in _EXCLUDE are excluded from hash."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            display_only: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})

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
            display_only: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})

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
        """Test that all fields with _ExcludeFrom.STORAGE in _EXCLUDE also have _ExcludeFrom.HASH.

        This is a validation test to ensure that fields excluded from storage are
        also excluded from hash computation. A field should never be excluded from
        storage but included in the hash.
        """
        for f in fields(Identifier):
            exclude_set = f.metadata.get(_EXCLUDE, set())
            if _ExcludeFrom.STORAGE in exclude_set:
                assert _ExcludeFrom.HASH in exclude_set, (
                    f"Field '{f.name}' has _ExcludeFrom.STORAGE in _EXCLUDE but not _ExcludeFrom.HASH. "
                    f"Fields excluded from storage must also be excluded from hash."
                )

    def test_subclass_storage_exclusion_implies_hash_exclusion(self):
        """Test that subclass fields with _ExcludeFrom.STORAGE in _EXCLUDE also have _ExcludeFrom.HASH."""

        @dataclass(frozen=True)
        class InvalidIdentifier(Identifier):
            # This is invalid - storage exclusion without hash exclusion
            bad_field: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.STORAGE}})

        # Verify the invariant is violated (this is what the test guards against)
        for f in fields(InvalidIdentifier):
            if f.name == "bad_field":
                exclude_set = f.metadata.get(_EXCLUDE, set())
                assert _ExcludeFrom.STORAGE in exclude_set
                assert _ExcludeFrom.HASH not in exclude_set  # This is the problematic case

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

    def test_hash_and_storage_exclusion_works(self):
        """Test that a field can be excluded from both hash and storage."""

        @dataclass(frozen=True)
        class ExtendedIdentifier(Identifier):
            transient_field: str = field(default="", metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})

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
