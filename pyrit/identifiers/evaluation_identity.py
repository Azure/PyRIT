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
* ``ScorerEvaluationIdentity`` — scorer-domain concrete subclass.
* ``AtomicAttackEvaluationIdentity`` — attack-domain concrete subclass.
* ``compute_attack_eval_hash`` — convenience wrapper for attacks.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Optional

from pyrit.identifiers.component_identifier import ComponentIdentifier, config_hash


def _build_eval_dict(
    identifier: ComponentIdentifier,
    *,
    behavioral_child_params: dict[str, frozenset[str]],
    ignored_child_keys: frozenset[str] = frozenset(),
    param_allowlist: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Build a filtered dictionary for eval-hash computation.

    Includes only behavioral parameters. For child components whose names appear
    as keys in ``behavioral_child_params``, only the mapped param frozenset is
    kept (stripping operational params like endpoint, max_requests_per_minute).
    Children in ``ignored_child_keys`` are completely excluded from the hash.
    All other children receive full eval treatment recursively.

    Args:
        identifier (ComponentIdentifier): The component identity to process.
        behavioral_child_params (dict[str, frozenset[str]]): Mapping of target
            child names to param allowlists. Children whose names appear as keys
            are filtered; all other children receive full recursive treatment.
        ignored_child_keys (frozenset[str]): Child names to completely exclude
            from the eval hash.
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
            if name in ignored_child_keys:
                continue
            child_list = identifier.get_child_list(name)
            if name in behavioral_child_params:
                # Targets: filter to child-specific behavioral params only
                child_allowlist = behavioral_child_params[name]
                hashes = [
                    config_hash(
                        _build_eval_dict(
                            c,
                            behavioral_child_params=behavioral_child_params,
                            ignored_child_keys=ignored_child_keys,
                            param_allowlist=child_allowlist,
                        )
                    )
                    for c in child_list
                ]
            else:
                # Non-targets (e.g., converters, scorers): full eval treatment, recurse without param filtering
                hashes = [
                    config_hash(
                        _build_eval_dict(
                            c,
                            behavioral_child_params=behavioral_child_params,
                            ignored_child_keys=ignored_child_keys,
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
    behavioral_child_params: dict[str, frozenset[str]],
    ignored_child_keys: frozenset[str] = frozenset(),
) -> str:
    """
    Compute a behavioral equivalence hash for evaluation grouping.

    Unlike ``ComponentIdentifier.hash`` (which includes all params of self and
    children), the eval hash filters child components that are "targets" to only
    their behavioral params (e.g., model_name, temperature, top_p), stripping
    operational params like endpoint or max_requests_per_minute. Children in
    ``ignored_child_keys`` are completely excluded. This ensures the same logical
    configuration on different deployments produces the same eval hash.

    Non-target children (e.g., sub-scorers, converters) receive full recursive eval
    treatment.

    When ``behavioral_child_params`` is empty and ``ignored_child_keys`` is empty,
    no filtering occurs and the result equals ``identifier.hash``.

    Args:
        identifier (ComponentIdentifier): The component identity to compute the hash for.
        behavioral_child_params (dict[str, frozenset[str]]): Mapping of target child
            names to their param allowlists. Children whose names appear as keys are
            filtered to only the specified params; all other children are included
            fully.
        ignored_child_keys (frozenset[str]): Child names to completely exclude
            from the eval hash.

    Returns:
        str: A hex-encoded SHA256 hash suitable for eval registry keying.
    """
    if not behavioral_child_params and not ignored_child_keys:
        return identifier.hash

    eval_dict = _build_eval_dict(
        identifier,
        behavioral_child_params=behavioral_child_params,
        ignored_child_keys=ignored_child_keys,
    )
    return config_hash(eval_dict)


class EvaluationIdentity(ABC):
    """
    Wraps a ``ComponentIdentifier`` with domain-specific eval-hash configuration.

    Subclasses must set the ``ClassVar`` values:

    * ``BEHAVIORAL_CHILD_PARAMS`` — mapping of target child names to param
      allowlists. Children whose names appear as keys have their params filtered
      to only the specified set; all other children are included fully.
    * ``IGNORED_CHILD_KEYS`` — child names to completely exclude from the
      eval hash (default: empty).

    The concrete ``eval_hash`` property delegates to the module-level
    ``compute_eval_hash`` free function.
    """

    BEHAVIORAL_CHILD_PARAMS: ClassVar[dict[str, frozenset[str]]]
    IGNORED_CHILD_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, identifier: ComponentIdentifier) -> None:
        """Wrap a ComponentIdentifier and eagerly compute its eval hash."""
        self._identifier = identifier
        self._eval_hash = compute_eval_hash(
            identifier,
            behavioral_child_params=self.BEHAVIORAL_CHILD_PARAMS,
            ignored_child_keys=self.IGNORED_CHILD_KEYS,
        )

    @property
    def identifier(self) -> ComponentIdentifier:
        """The underlying component identity."""
        return self._identifier

    @property
    def eval_hash(self) -> str:
        """Behavioral equivalence hash for evaluation grouping."""
        return self._eval_hash


class ScorerEvaluationIdentity(EvaluationIdentity):
    """
    Evaluation identity for scorers.

    Target children (``prompt_target``, ``converter_target``) are filtered to
    behavioral params only (``model_name``, ``temperature``, ``top_p``), so the
    same scorer configuration on different deployments produces the same eval hash.
    """

    BEHAVIORAL_CHILD_PARAMS: ClassVar[dict[str, frozenset[str]]] = {
        "prompt_target": frozenset({"model_name", "temperature", "top_p"}),
        "converter_target": frozenset({"model_name", "temperature", "top_p"}),
    }


class AtomicAttackEvaluationIdentity(EvaluationIdentity):
    """
    Evaluation identity for atomic attacks.

    The ``objective_target`` child is filtered to only ``temperature``.
    The ``adversarial_chat`` child is filtered to ``model_name``,
    ``temperature``, and ``top_p``.
    The ``objective_scorer`` child is completely excluded.

    Non-target children (e.g., ``request_converters``, ``response_converters``,
    and seed identifiers) receive full recursive eval treatment, meaning they
    fully contribute to the hash.
    """

    BEHAVIORAL_CHILD_PARAMS: ClassVar[dict[str, frozenset[str]]] = {
        "objective_target": frozenset({"temperature"}),
        "adversarial_chat": frozenset({"model_name", "temperature", "top_p"}),
    }
    IGNORED_CHILD_KEYS: ClassVar[frozenset[str]] = frozenset({"objective_scorer"})


def compute_attack_eval_hash(identifier: ComponentIdentifier) -> str:
    """
    Compute a behavioral equivalence hash for attack evaluation grouping.

    Convenience wrapper around ``compute_eval_hash`` with attack-specific
    constants.  The ``objective_scorer`` is excluded. For the
    ``objective_target`` child, only temperature is included. For
    ``adversarial_chat``, model_name, temperature, and top_p are included.
    For all other children (converters, seeds), all params are included.

    Args:
        identifier (ComponentIdentifier): The atomic attack's composite identity.

    Returns:
        str: A hash suitable for evaluation registry keying.
    """
    return compute_eval_hash(
        identifier,
        behavioral_child_params=AtomicAttackEvaluationIdentity.BEHAVIORAL_CHILD_PARAMS,
        ignored_child_keys=AtomicAttackEvaluationIdentity.IGNORED_CHILD_KEYS,
    )
