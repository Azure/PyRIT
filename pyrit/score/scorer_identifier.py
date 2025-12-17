# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pyrit


@dataclass
class ScorerIdentifier:
    """
    Configuration class for Scorers.

    This class encapsulates the modifiable parameters that can be used to create a complete scoring configuration.
    These parameters can be modified, and configurations can be compared to each other via scorer evaluations.
    """

    type: str
    version: int
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None
    sub_identifier: Optional[List["ScorerIdentifier"]] = None
    model_info: Optional[Dict[str, Any]] = None
    score_aggregator: Optional[str] = None
    scorer_specific_params: Optional[Dict[str, Any]] = None
    pyrit_version: str = pyrit.__version__

    def compute_hash(self) -> str:
        """
        Compute a hash representing the current configuration.

        Uses to_compact_dict() to get the compacted representation, then hashes it.

        Returns:
            str: A hash string representing the configuration.
        """
        config_dict = self.to_compact_dict()

        # Sort keys to ensure deterministic ordering and encode as JSON
        config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))

        hasher = hashlib.sha256()
        hasher.update(config_json.encode("utf-8"))
        return hasher.hexdigest()

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        Convert the ScorerIdentifier to a compact dictionary for storage.

        Long prompts (>100 characters) are hashed to sha256:{hash[:16]} format.
        Nested sub_identifiers are recursively compacted.
        Uses __type__ key for consistency with PyRIT conventions.

        Returns:
            Dict[str, Any]: A compact dictionary representation.
        """
        # Hash system_prompt_template if longer than 100 characters
        sys_prompt = self.system_prompt_template
        if sys_prompt and len(sys_prompt) > 100:
            sys_prompt = f"sha256:{hashlib.sha256(sys_prompt.encode()).hexdigest()[:16]}"

        # Hash user_prompt_template if longer than 100 characters
        user_prompt = self.user_prompt_template
        if user_prompt and len(user_prompt) > 100:
            user_prompt = f"sha256:{hashlib.sha256(user_prompt.encode()).hexdigest()[:16]}"

        # Recursively compact sub_identifiers
        sub_id_serialized: Any = None
        if self.sub_identifier is not None:
            sub_id_serialized = [si.to_compact_dict() for si in self.sub_identifier]

        return {
            "__type__": self.type,
            "version": self.version,
            "system_prompt_template": sys_prompt,
            "user_prompt_template": user_prompt,
            "sub_identifier": sub_id_serialized,
            "model_info": self.model_info,
            "score_aggregator": self.score_aggregator,
            "scorer_specific_params": self.scorer_specific_params,
            "pyrit_version": self.pyrit_version,
        }
