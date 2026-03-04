# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Evaluation identity and eval-hash computation.

This module provides:

* ``_build_eval_dict`` — builds a filtered dict for eval-hash computation.
* ``compute_eval_hash`` — free function that computes a behavioral equivalence
  hash from a ``ComponentIdentifier``.
* ``EvaluationIdentity`` — abstract base that wraps a ``ComponentIdentifier``
  with domain-specific eval-hash configuration.  Concrete subclasses declare
  *which* children are targets and *which* params are behavioral via two
  ``ClassVar`` frozensets.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Optional

from pyrit.identifiers.component_identifier import ComponentIdentifier, config_hash


def _build_eval_dict(
    identifier: ComponentIdentifier,
    *,
    target_child_keys: frozenset[str],
    behavioral_child_params: frozenset[str],
    param_allowlist: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Build a filtered dictionary for eval-hash computation.

    Includes only behavioral parameters. For child components whose names appear
    in ``target_child_keys``, only params in ``behavioral_child_params`` are kept
    (stripping operational params like endpoint, max_requests_per_minute).
    Non-target children receive full eval treatment recursively.

    Args:
        identifier (ComponentIdentifier): The component identity to process.
        target_child_keys (frozenset[str]): Child names that are targets
            (e.g., ``{"prompt_target", "converter_target"}``).
        behavioral_child_params (frozenset[str]): Param allowlist applied to
            target children (e.g., ``{"model_name", "temperature", "top_p"}``).
        param_allowlist (Optional[frozenset[str]]): If provided, only include
            params whose keys are in the allowlist. If None, include all params.

    Returns:
        dict[str, Any]: The filtered dictionary suitable for hashing.
    """
    eval_dict: dict[str, Any] = {
        ComponentIdentifier.KEY_CLASS_NAME: identifier.class_name,
        ComponentIdentifier.KEY_CLASS_MODULE: identifier.class_module,
    }

    eval_dict.update(
        {
            key: value
            for key, value in sorted(identifier.params.items())
            if value is not None and (param_allowlist is None or key in param_allowlist)
        }
    )

    if identifier.children:
        eval_children: dict[str, Any] = {}
        for name in sorted(identifier.children):
            child_list = identifier.get_child_list(name)
            if name in target_child_keys:
                # Targets: filter to behavioral params only
                hashes = [
                    config_hash(
                        _build_eval_dict(
                            c,
                            target_child_keys=target_child_keys,
                            behavioral_child_params=behavioral_child_params,
                            param_allowlist=behavioral_child_params,
                        )
                    )
                    for c in child_list
                ]
            else:
                # Non-targets (e.g., sub-scorers): full eval treatment, recurse without param filtering
                hashes = [
                    config_hash(
                        _build_eval_dict(
                            c,
                            target_child_keys=target_child_keys,
                            behavioral_child_params=behavioral_child_params,
                        )
                    )
                    for c in child_list
                ]
            eval_children[name] = hashes[0] if len(hashes) == 1 else hashes
        if eval_children:
            eval_dict["children"] = eval_children

    return eval_dict


def compute_eval_hash(
    identifier: ComponentIdentifier,
    *,
    target_child_keys: frozenset[str],
    behavioral_child_params: frozenset[str],
) -> str:
    """
    Compute a behavioral equivalence hash for evaluation grouping.

    Unlike ``ComponentIdentifier.hash`` (which includes all params of self and
    children), the eval hash filters child components that are "targets" to only
    their behavioral params (e.g., model_name, temperature, top_p), stripping
    operational params like endpoint or max_requests_per_minute. This ensures the
    same logical configuration on different deployments produces the same eval hash.

    Non-target children (e.g., sub-scorers) receive full recursive eval treatment.

    When ``target_child_keys`` is empty, no child filtering occurs and the result
    equals ``identifier.hash``.

    Args:
        identifier (ComponentIdentifier): The component identity to compute the hash for.
        target_child_keys (frozenset[str]): Child names that are targets
            (e.g., ``{"prompt_target", "converter_target"}``).
        behavioral_child_params (frozenset[str]): Param allowlist for target children
            (e.g., ``{"model_name", "temperature", "top_p"}``).

    Returns:
        str: A hex-encoded SHA256 hash suitable for eval registry keying.
    """
    if not target_child_keys:
        return identifier.hash

    eval_dict = _build_eval_dict(
        identifier,
        target_child_keys=target_child_keys,
        behavioral_child_params=behavioral_child_params,
    )
    return config_hash(eval_dict)


class EvaluationIdentity(ABC):
    """
    Wraps a ``ComponentIdentifier`` with domain-specific eval-hash configuration.

    Subclasses must set the two ``ClassVar`` frozensets:

    * ``TARGET_CHILD_KEYS`` — child names whose operational params should be
      stripped (e.g., ``{"prompt_target", "converter_target"}``).
    * ``BEHAVIORAL_CHILD_PARAMS`` — param allowlist applied to those target
      children (e.g., ``{"model_name", "temperature", "top_p"}``).

    The concrete ``eval_hash`` property delegates to the module-level
    ``compute_eval_hash`` free function.
    """

    TARGET_CHILD_KEYS: ClassVar[frozenset[str]]
    BEHAVIORAL_CHILD_PARAMS: ClassVar[frozenset[str]]

    def __init__(self, identifier: ComponentIdentifier) -> None:
        """Wrap a ComponentIdentifier and eagerly compute its eval hash."""
        self._identifier = identifier
        self._eval_hash = compute_eval_hash(
            identifier,
            target_child_keys=self.TARGET_CHILD_KEYS,
            behavioral_child_params=self.BEHAVIORAL_CHILD_PARAMS,
        )

    @property
    def identifier(self) -> ComponentIdentifier:
        """The underlying component identity."""
        return self._identifier

    @property
    def eval_hash(self) -> str:
        """Behavioral equivalence hash for evaluation grouping."""
        return self._eval_hash
