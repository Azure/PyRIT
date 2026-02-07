# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, cast

from pyrit.identifiers.converter_identifier import ConverterIdentifier
from pyrit.identifiers.identifier import Identifier
from pyrit.identifiers.scorer_identifier import ScorerIdentifier
from pyrit.identifiers.target_identifier import TargetIdentifier


@dataclass(frozen=True)
class AttackIdentifier(Identifier):
    """
    Typed identifier for an attack strategy instance.

    Captures the configuration that makes one attack strategy meaningfully
    different from another: the objective target, optional scorer, and converter
    pipeline. These do not change between calls to ``execute_async``.
    """

    objective_target_identifier: Optional[TargetIdentifier] = None
    objective_scorer_identifier: Optional[ScorerIdentifier] = None
    request_converter_identifiers: Optional[List[ConverterIdentifier]] = None

    # Additional attack-specific params for subclass flexibility
    attack_specific_params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls: Type["AttackIdentifier"], data: dict[str, Any]) -> "AttackIdentifier":
        """
        Deserialize an AttackIdentifier from a dictionary.

        Handles nested sub-identifiers (target, scorer, converters) by
        recursively calling their own ``from_dict`` implementations.

        Args:
            data: Dictionary containing the serialized identifier fields.

        Returns:
            AttackIdentifier: The deserialized identifier.
        """
        data = dict(data)

        if "objective_target_identifier" in data and isinstance(data["objective_target_identifier"], dict):
            data["objective_target_identifier"] = TargetIdentifier.from_dict(data["objective_target_identifier"])

        if "objective_scorer_identifier" in data and isinstance(data["objective_scorer_identifier"], dict):
            data["objective_scorer_identifier"] = ScorerIdentifier.from_dict(data["objective_scorer_identifier"])

        if "request_converter_identifiers" in data and data["request_converter_identifiers"] is not None:
            data["request_converter_identifiers"] = [
                ConverterIdentifier.from_dict(c) if isinstance(c, dict) else c
                for c in data["request_converter_identifiers"]
            ]

        result = Identifier.from_dict.__func__(cls, data)  # type: ignore[attr-defined]
        return cast(AttackIdentifier, result)
