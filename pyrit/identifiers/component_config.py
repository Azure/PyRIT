# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Component configuration and identity system for PyRIT.

A ComponentConfig is an immutable snapshot of a component's behavioral configuration,
serving as both its identity and its storable representation.

Design principles:
    1. The config dict IS the identity — no wrapper hierarchy needed.
    2. Hash is content-addressed from behavioral params only.
    3. Children carry their own hashes — compositional by default.
    4. Adding optional params with None default is backward-compatible (None values excluded).

ComponentConfig also satisfies the registry metadata contract (has class_name, class_module,
snake_class_name), so it can be used directly as metadata in instance registries like
ScorerRegistry without a separate wrapper.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union

import pyrit
from pyrit.identifiers.class_name_utils import class_name_to_snake_case

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def config_hash(config_dict: Dict[str, Any]) -> str:
    """
    Compute a deterministic SHA256 hash from a config dictionary.

    This is the single source of truth for identity hashing across the entire
    system. The dict is serialized with sorted keys and compact separators to
    ensure determinism.

    Args:
        config_dict (Dict[str, Any]): A JSON-serializable dictionary.

    Returns:
        str: Hex-encoded SHA256 hash string.

    Raises:
        TypeError: If config_dict contains values that are not JSON-serializable.
    """
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_hash_dict(
    *,
    class_name: str,
    class_module: str,
    params: Dict[str, Any],
    children: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the canonical dictionary used for hash computation.

    Children are represented by their hashes, not their full config.
    A parent's hash changes when a child's behavioral config changes,
    but the parent doesn't need to understand the child's internal structure.

    Args:
        class_name (str): The component's class name.
        class_module (str): The component's module path.
        params (Dict[str, Any]): Behavioral parameters (non-None values only).
        children (Dict[str, Any]): Child name to ComponentConfig or list of ComponentConfig.

    Returns:
        Dict[str, Any]: The canonical dictionary for hashing.
    """
    hash_dict: Dict[str, Any] = {
        ComponentConfig.KEY_CLASS_NAME: class_name,
        ComponentConfig.KEY_CLASS_MODULE: class_module,
    }

    # Only include non-None params — adding an optional param with None default
    # won't change existing hashes, making the schema backward-compatible.
    for key, value in sorted(params.items()):
        if value is not None:
            hash_dict[key] = value

    # Children contribute their hashes, not their full structure.
    if children:
        children_hashes: Dict[str, Any] = {}
        for name, child in sorted(children.items()):
            if isinstance(child, ComponentConfig):
                children_hashes[name] = child.hash
            elif isinstance(child, list):
                children_hashes[name] = [c.hash for c in child if isinstance(c, ComponentConfig)]
        if children_hashes:
            hash_dict[ComponentConfig.KEY_CHILDREN] = children_hashes

    return hash_dict


# ---------------------------------------------------------------------------
# ComponentConfig — the frozen identity snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentConfig:
    """
    Immutable snapshot of a component's behavioral configuration.

    A single type for all component identity — scorers, targets, converters, and
    any future component types all produce a ComponentConfig with their relevant
    params and children.

    The hash is content-addressed: two ComponentConfigs with the same class, params,
    and children produce the same hash. This enables deterministic metrics lookup,
    DB deduplication, and registry keying.

    Also usable as registry metadata for instance registries (e.g., ScorerRegistry)
    because it exposes ``class_name``, ``snake_class_name``, and ``unique_name``.

    Attributes:
        class_name (str): Python class name (e.g., "SelfAskScaleScorer").
        class_module (str): Full module path (e.g., "pyrit.score.self_ask_scale_scorer").
        params (Dict[str, Any]): Behavioral parameters that affect output.
        children (Dict[str, Union[ComponentConfig, List[ComponentConfig]]]): Named
            child configs for compositional identity (e.g., a scorer's target).
        hash (str): Content-addressed SHA256 hash computed from class, params, and children.
        pyrit_version (str): Version tag for storage. Not included in hash.
    """

    # -------------------------------------------------------------------
    # Serialization key constants
    # -------------------------------------------------------------------

    KEY_CLASS_NAME: ClassVar[str] = "class_name"
    KEY_CLASS_MODULE: ClassVar[str] = "class_module"
    KEY_HASH: ClassVar[str] = "hash"
    KEY_PYRIT_VERSION: ClassVar[str] = "pyrit_version"
    KEY_CHILDREN: ClassVar[str] = "children"
    LEGACY_KEY_TYPE: ClassVar[str] = "__type__"
    LEGACY_KEY_MODULE: ClassVar[str] = "__module__"

    # -------------------------------------------------------------------
    # Fields
    # -------------------------------------------------------------------

    class_name: str
    class_module: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: Dict[str, Union[ComponentConfig, List[ComponentConfig]]] = field(default_factory=dict)
    hash: str = field(init=False, compare=False)
    pyrit_version: str = field(default_factory=lambda: pyrit.__version__, compare=False)

    def __post_init__(self) -> None:
        """Compute the content-addressed hash at creation time."""
        hash_dict = _build_hash_dict(
            class_name=self.class_name,
            class_module=self.class_module,
            params=self.params,
            children=self.children,
        )
        object.__setattr__(self, "hash", config_hash(hash_dict))

    # -------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------

    @property
    def short_hash(self) -> str:
        """
        Return the first 8 characters of the hash for display and logging.

        This truncated hash provides sufficient uniqueness for human-readable
        identification while keeping output concise. Used in string representations,
        log messages, and registry keys.

        Returns:
            str: First 8 hex characters of the SHA256 hash.
        """
        return self.hash[:8]

    @property
    def snake_class_name(self) -> str:
        """
        Snake_case version of class_name (e.g., "self_ask_scale_scorer").

        Used by registries for key derivation and CLI formatting.
        """
        return class_name_to_snake_case(self.class_name)

    @property
    def unique_name(self) -> str:
        """
        Globally unique display name: ``snake_class_name::short_hash``.

        Used as the default registration key in instance registries
        (e.g., "self_ask_scale_scorer::a1b2c3d4").
        """
        return f"{self.snake_class_name}::{self.short_hash}"

    # -------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------

    @classmethod
    def of(
        cls,
        obj: object,
        *,
        params: Optional[Dict[str, Any]] = None,
        children: Optional[Dict[str, Union[ComponentConfig, List[ComponentConfig]]]] = None,
    ) -> ComponentConfig:
        """
        Build a ComponentConfig from a live object instance.

        This factory method extracts class_name and class_module from the object's
        type automatically, making it the preferred way to create configs in
        component implementations. None-valued params and children are filtered out
        to ensure backward-compatible hashing.

        Args:
            obj (object): The live component instance whose type info will be captured.
            params (Optional[Dict[str, Any]]): Behavioral parameters that affect the
                component's output. Only include params that change behavior — exclude
                operational settings like rate limits, retry counts, or logging config.
            children (Optional[Dict[str, Union[ComponentConfig, List[ComponentConfig]]]]):
                Named child component configs. Use for compositional components like
                scorers that wrap other scorers or targets that chain converters.

        Returns:
            ComponentConfig: The frozen config snapshot with computed hash.
        """
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        clean_children = {k: v for k, v in (children or {}).items() if v is not None}

        return cls(
            class_name=obj.__class__.__name__,
            class_module=obj.__class__.__module__,
            params=clean_params,
            children=clean_children,
        )

    # -------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------

    @classmethod
    def normalize(cls, value: Union[ComponentConfig, Dict[str, Any]]) -> ComponentConfig:
        """
        Normalize a value to a ComponentConfig instance.

        Accepts either an existing ComponentConfig (returned as-is) or a dict
        (reconstructed via from_dict). This supports code paths that may receive
        either typed configs or raw dicts from database storage.

        Args:
            value (Union[ComponentConfig, Dict[str, Any]]): A ComponentConfig or
                a dictionary representation.

        Returns:
            ComponentConfig: The normalized config instance.

        Raises:
            TypeError: If value is neither a ComponentConfig nor a dict.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        raise TypeError(f"Expected ComponentConfig or dict, got {type(value).__name__}")

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-compatible dictionary for DB/JSONL storage.

        Produces a flat structure where params are inlined at the top level alongside
        class_name, class_module, hash, and pyrit_version. This maintains backward
        compatibility with existing DB queries that access params directly
        (e.g., ``scorer_class_identifier->>'scorer_type'``).

        Children are recursively serialized into a nested "children" key.

        Returns:
            Dict[str, Any]: JSON-serializable dictionary suitable for database storage
                or JSONL export.
        """
        result: Dict[str, Any] = {
            self.KEY_CLASS_NAME: self.class_name,
            self.KEY_CLASS_MODULE: self.class_module,
            self.KEY_HASH: self.hash,
            self.KEY_PYRIT_VERSION: self.pyrit_version,
        }

        for key, value in self.params.items():
            result[key] = value

        if self.children:
            serialized_children: Dict[str, Any] = {}
            for name, child in self.children.items():
                if isinstance(child, ComponentConfig):
                    serialized_children[name] = child.to_dict()
                elif isinstance(child, list):
                    serialized_children[name] = [c.to_dict() for c in child]
            result[self.KEY_CHILDREN] = serialized_children

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComponentConfig:
        """
        Deserialize from a stored dictionary.

        Reconstructs a ComponentConfig from data previously saved via to_dict().
        Handles both the current format (``class_name``/``class_module``) and legacy
        format (``__type__``/``__module__``) for backward compatibility with
        older database records.

        The hash is always recomputed from the reconstructed params and children.
        If the stored hash differs from the computed hash, a warning is logged
        indicating possible schema drift.

        Args:
            data (Dict[str, Any]): Dictionary from DB/JSONL storage. The original
                dict is not mutated; a copy is made internally.

        Returns:
            ComponentConfig: Reconstructed config with freshly computed hash.
        """
        data = dict(data)  # Don't mutate the input

        # Handle legacy key mappings
        class_name = (
            data.pop(cls.KEY_CLASS_NAME, None) or data.pop(cls.LEGACY_KEY_TYPE, None) or "Unknown"
        )
        class_module = (
            data.pop(cls.KEY_CLASS_MODULE, None) or data.pop(cls.LEGACY_KEY_MODULE, None) or "unknown"
        )

        stored_hash = data.pop(cls.KEY_HASH, None)
        pyrit_version = data.pop(cls.KEY_PYRIT_VERSION, pyrit.__version__)

        # Reconstruct children
        children: Dict[str, Union[ComponentConfig, List[ComponentConfig]]] = {}
        raw_children = data.pop(cls.KEY_CHILDREN, None)
        if raw_children and isinstance(raw_children, dict):
            for name, child_data in raw_children.items():
                if isinstance(child_data, dict):
                    children[name] = cls.from_dict(child_data)
                elif isinstance(child_data, list):
                    children[name] = [cls.from_dict(c) for c in child_data if isinstance(c, dict)]

        # Everything remaining is a param
        params = data

        config = cls(
            class_name=class_name,
            class_module=class_module,
            params=params,
            children=children,
            pyrit_version=pyrit_version,
        )

        if stored_hash and config.hash != stored_hash:
            logger.warning(
                f"Hash mismatch for {class_name}: stored={stored_hash[:8]}, "
                f"computed={config.short_hash}. Schema may have changed."
            )

        return config

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Format: ``ClassName::abcd1234`` (class name followed by short hash).
        Suitable for log messages and user-facing output.

        Returns:
            str: Human-readable identifier string.
        """
        return f"{self.class_name}::{self.short_hash}"

    def __repr__(self) -> str:
        """
        Return a detailed representation for debugging.

        Includes class name, all params, children references, and the short hash.
        Useful for inspecting config contents in debuggers or REPL sessions.

        Returns:
            str: Detailed debug string showing all config components.
        """
        params_str = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        children_str = ", ".join(f"{k}={v}" for k, v in sorted(self.children.items()))
        parts = [f"class={self.class_name}"]
        if params_str:
            parts.append(f"params=({params_str})")
        if children_str:
            parts.append(f"children=({children_str})")
        parts.append(f"hash={self.short_hash}")
        return f"ComponentConfig({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Configurable — the ABC components implement
# ---------------------------------------------------------------------------


class Configurable(ABC):
    """
    Abstract base class for components that describe their behavioral configuration.

    Components implement ``_build_config()`` to return a frozen ComponentConfig
    snapshot. The config is built lazily on first access and cached for the
    component's lifetime.
    """

    _config: Optional[ComponentConfig] = None

    @abstractmethod
    def _build_config(self) -> ComponentConfig:
        """
        Build the behavioral configuration for this component.

        Only include params that affect the component's behavior/output.
        Exclude operational params (rate limits, retry config, logging settings).

        Returns:
            ComponentConfig: The frozen configuration snapshot.
        """
        ...

    def get_config(self) -> ComponentConfig:
        """
        Get the component's configuration, building it lazily on first access.

        The config is computed once via _build_config() and then cached for
        subsequent calls. This ensures consistent identity throughout the
        component's lifetime while deferring computation until actually needed.

        Note:
            Not thread-safe. If thread safety is required, subclasses should
            implement appropriate synchronization.

        Returns:
            ComponentConfig: The frozen configuration snapshot representing
                this component's behavioral identity.
        """
        if self._config is None:
            self._config = self._build_config()
        return self._config