# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
import json
from dataclasses import Field, asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Literal, Type, TypeVar

import pyrit
from pyrit.common.deprecation import print_deprecation_message
from pyrit.identifiers.class_name_utils import class_name_to_snake_case

IdentifierType = Literal["class", "instance"]


class _ExcludeFrom(Enum):
    """
    Enum specifying what a field should be excluded from.

    Used as values in the _EXCLUDE metadata set for dataclass fields.

    Values:
        HASH: Exclude the field from hash computation (field is still stored).
        STORAGE: Exclude the field from storage (implies HASH - field is also excluded from hash).
    """

    HASH = "hash"
    STORAGE = "storage"


# Metadata keys for field configuration
# _EXCLUDE is a metadata key whose value is a set of _ExcludeFrom enum values.
# Examples:
#   field(metadata={_EXCLUDE: {_ExcludeFrom.HASH}})  # Stored but not hashed
#   field(metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})  # Not stored and not hashed
_EXCLUDE = "exclude"
_MAX_STORAGE_LENGTH = "max_storage_length"


def _is_excluded_from_hash(f: Field[Any]) -> bool:
    """
    Check if a field should be excluded from hash computation.

    A field is excluded from hash if it has _EXCLUDE metadata containing
    _ExcludeFrom.HASH or _ExcludeFrom.STORAGE.

    Args:
        f: A dataclass field object.

    Returns:
        True if the field should be excluded from hash computation.
    """
    exclude_set = f.metadata.get(_EXCLUDE, set())
    return _ExcludeFrom.HASH in exclude_set or _ExcludeFrom.STORAGE in exclude_set


def _is_excluded_from_storage(f: Field[Any]) -> bool:
    """
    Check if a field should be excluded from storage.

    A field is excluded from storage if it has _EXCLUDE metadata containing _ExcludeFrom.STORAGE.

    Args:
        f: A dataclass field object.

    Returns:
        True if the field should be excluded from storage.
    """
    exclude_set = f.metadata.get(_EXCLUDE, set())
    return _ExcludeFrom.STORAGE in exclude_set


T = TypeVar("T", bound="Identifier")


@dataclass(frozen=True)
class Identifier:
    """
    Base dataclass for identifying PyRIT components.

    This frozen dataclass provides a stable identifier for registry items,
    targets, scorers, attacks, converters, and other components. The hash is computed at creation
    time from the core fields and remains constant.

    This class serves as:
    1. Base for registry metadata (replacing RegistryItemMetadata)
    2. Future replacement for get_identifier() dict patterns

    All component-specific identifier types should extend this with additional fields.
    """

    class_name: str  # The actual class name, equivalent to __type__ (e.g., "SelfAskRefusalScorer")
    class_module: str  # The module path, equivalent to __module__ (e.g., "pyrit.score.self_ask_refusal_scorer")

    # Fields excluded from storage and hash
    class_description: str = field(metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})
    identifier_type: IdentifierType = field(metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})

    # Auto-computed fields
    snake_class_name: str = field(init=False, metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}})
    hash: str | None = field(default=None, compare=False, kw_only=True, metadata={_EXCLUDE: {_ExcludeFrom.HASH}})
    unique_name: str = field(init=False, metadata={_EXCLUDE: {_ExcludeFrom.HASH}})  # {full_snake_case}::{hash[:8]}

    # Version field - stored but not hashed (allows version tracking without affecting identity)
    pyrit_version: str = field(
        default_factory=lambda: pyrit.__version__, kw_only=True, metadata={_EXCLUDE: {_ExcludeFrom.HASH}}
    )

    def __post_init__(self) -> None:
        """Compute derived fields: snake_class_name, hash, and unique_name."""
        # Use object.__setattr__ since this is a frozen dataclass
        # 1. Compute snake_class_name
        object.__setattr__(self, "snake_class_name", class_name_to_snake_case(self.class_name))
        # 2. Compute hash only if not already provided (e.g., from from_dict)
        computed_hash = self.hash if self.hash is not None else self._compute_hash()
        object.__setattr__(self, "hash", computed_hash)
        # 3. Compute unique_name: full snake_case :: hash prefix
        full_snake = class_name_to_snake_case(self.class_name)
        object.__setattr__(self, "unique_name", f"{full_snake}::{computed_hash[:8]}")

    def _compute_hash(self) -> str:
        """
        Compute a stable SHA256 hash from identifier fields not excluded from hashing.

        Fields are excluded from hash computation if they have:
        metadata={_EXCLUDE: {_ExcludeFrom.HASH}} or metadata={_EXCLUDE: {_ExcludeFrom.HASH, _ExcludeFrom.STORAGE}}

        Returns:
            A hex string of the SHA256 hash.
        """
        hashable_dict: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in fields(self) if not _is_excluded_from_hash(f)
        }
        config_json = json.dumps(hashable_dict, sort_keys=True, separators=(",", ":"), default=_dataclass_encoder)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """
        Return only fields suitable for DB storage.

        Fields with max_storage_length metadata are truncated to show the first
        N characters followed by the field's hash, formatted as:
        "<first N chars>... [sha256:<hash[:16]>]"

        Nested Identifier objects are recursively serialized to dicts.

        Returns:
            dict[str, Any]: A dictionary containing the storable fields.
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            if _is_excluded_from_storage(f):
                continue
            value = getattr(self, f.name)
            max_len = f.metadata.get(_MAX_STORAGE_LENGTH)
            if max_len is not None and isinstance(value, str) and len(value) > max_len:
                truncated = value[:max_len]
                field_hash = hashlib.sha256(value.encode()).hexdigest()[:16]
                value = f"{truncated}... [sha256:{field_hash}]"
            # Recursively serialize nested Identifier objects
            elif isinstance(value, Identifier):
                value = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], Identifier):
                value = [item.to_dict() for item in value]
            # Exclude None and empty values
            if value is None or value == "" or value == [] or value == {}:
                continue
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """
        Create an Identifier from a dictionary (e.g., retrieved from database).

        Note:
            For fields with max_storage_length, stored values may be truncated
            strings like "<first N chars>... [sha256:<hash>]". If a 'hash' key is
            present in the input dict, it will be preserved rather than recomputed,
            ensuring identity matching works correctly.

        Args:
            data: The dictionary representation.

        Returns:
            A new Identifier instance.
        """
        # Create a mutable copy
        data = dict(data)

        # Handle legacy key mappings for class_name
        if "class_name" not in data:
            if "__type__" in data:
                print_deprecation_message(
                    old_item="'__type__' key in Identifier dict",
                    new_item="'class_name' key",
                    removed_in="0.13.0",
                )
                data["class_name"] = data.pop("__type__")
            elif "type" in data:
                print_deprecation_message(
                    old_item="'type' key in Identifier dict",
                    new_item="'class_name' key",
                    removed_in="0.13.0",
                )
                data["class_name"] = data.pop("type")
            else:
                # Default for truly legacy data without any class identifier
                data["class_name"] = "Unknown"

        # Handle legacy key mapping for class_module
        if "class_module" not in data:
            if "__module__" in data:
                print_deprecation_message(
                    old_item="'__module__' key in Identifier dict",
                    new_item="'class_module' key",
                    removed_in="0.13.0",
                )
                data["class_module"] = data.pop("__module__")
            else:
                # Default for truly legacy data without module info
                data["class_module"] = "unknown"

        # Provide defaults for fields excluded from storage (not in stored dicts)
        if "class_description" not in data:
            data["class_description"] = ""
        if "identifier_type" not in data:
            data["identifier_type"] = "instance"

        # Get the set of valid field names for this class
        valid_fields = {f.name for f in fields(cls) if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    @classmethod
    def normalize(cls: Type[T], value: T | dict[str, Any]) -> T:
        """
        Normalize a value to an Identifier instance.

        This method handles conversion from legacy dict format to Identifier,
        emitting a deprecation warning when a dict is passed. Existing Identifier
        instances are returned as-is.

        Args:
            value: An Identifier instance or a dict (legacy format).

        Returns:
            The normalized Identifier instance.

        Raises:
            TypeError: If value is not an Identifier or dict.
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, dict):
            print_deprecation_message(
                old_item=f"dict for {cls.__name__}",
                new_item=cls.__name__,
                removed_in="0.14.0",
            )
            return cls.from_dict(value)

        raise TypeError(f"Expected {cls.__name__} or dict, got {type(value).__name__}")


def _dataclass_encoder(obj: Any) -> Any:
    """
    JSON encoder that handles dataclasses by converting them to dicts.

    Args:
        obj: The object to encode.

    Returns:
        Any: The dictionary representation of the dataclass.

    Raises:
        TypeError: If the object is not a dataclass instance.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
