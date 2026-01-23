# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal

IdentifierType = Literal["class", "instance"]


class Identifiable:
    """
    Abstract base class for objects that can provide an identifier dictionary.

    This is a legacy interface that will eventually be replaced by Identifier dataclass.
    Classes implementing this interface should return a dict describing their identity.
    """

    @abstractmethod
    def get_identifier(self) -> dict[str, str]:
        pass

    def __str__(self) -> str:
        return f"{self.get_identifier}"


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

    name: str  # The snake_case identifier name (e.g., "self_ask_refusal")
    class_name: str  # The actual class name, equivalent to __type__ (e.g., "SelfAskRefusalScorer")
    class_module: str  # The module path, equivalent to __module__ (e.g., "pyrit.score.self_ask_refusal_scorer")

    class_description: str = field(metadata={"exclude_from_storage": True})

    #  Whether this identifies a "class" or "instance"
    identifier_type: IdentifierType = field(metadata={"exclude_from_storage": True})
    hash: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Compute the identifier hash from core fields."""
        # Use object.__setattr__ since this is a frozen dataclass
        object.__setattr__(self, "hash", self._compute_hash())

    def _compute_hash(self) -> str:
        """
        Compute a stable SHA256 hash from storable identifier fields.

        Fields marked with metadata={"exclude_from_storage": True} and 'hash' itself
        are excluded from the hash computation.

        Returns:
            A hex string of the SHA256 hash.
        """
        hashable_dict: dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "hash" and not f.metadata.get("exclude_from_storage", False)
        }
        config_json = json.dumps(hashable_dict, sort_keys=True, separators=(",", ":"), default=_dataclass_encoder)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def to_storage_dict(self) -> dict[str, Any]:
        """Return only fields suitable for DB storage."""
        return {
            f.name: getattr(self, f.name) for f in fields(self) if not f.metadata.get("exclude_from_storage", False)
        }


def _dataclass_encoder(obj: Any) -> Any:
    """JSON encoder that handles dataclasses by converting them to dicts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
