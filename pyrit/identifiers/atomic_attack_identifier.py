# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Atomic attack identity builder functions.

Builds a composite ComponentIdentifier that uniquely identifies an attack run
by combining the attack strategy's identity with the general technique seed
identifiers from the dataset.

The composite identifier always has the same shape:
    class_name = "AtomicAttack"
    children["attack"] = attack strategy's ComponentIdentifier
    children["general_technique_seeds"] = list of seed ComponentIdentifiers
        (may be empty when no general technique seeds are present)
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from pyrit.identifiers.component_identifier import ComponentIdentifier, config_hash

if TYPE_CHECKING:
    from pyrit.models.seeds.seed import Seed
    from pyrit.models.seeds.seed_group import SeedGroup

logger = logging.getLogger(__name__)

# Child component params that affect attack behavior.
# Operational params (endpoint, max_requests_per_minute, etc.) are excluded
# so that the same model on different deployments shares cached eval results.
_BEHAVIORAL_CHILD_PARAMS = frozenset({"model_name", "temperature", "top_p"})
_TARGET_CHILD_KEYS = frozenset({"objective_target", "prompt_target", "converter_target"})

# Class metadata for the composite identifier
_ATOMIC_ATTACK_CLASS_NAME = "AtomicAttack"
_ATOMIC_ATTACK_CLASS_MODULE = "pyrit.scenario.core.atomic_attack"


def build_seed_identifier(seed: "Seed") -> ComponentIdentifier:
    """
    Build a ComponentIdentifier from a seed's behavioral properties.

    Captures the seed's content hash, dataset name, and class type so that
    different seeds produce different identifiers while the same seed content
    always produces the same identifier.

    Args:
        seed (Seed): The seed to build an identifier for.

    Returns:
        ComponentIdentifier: An identifier capturing the seed's behavioral properties.
    """
    params: dict[str, Any] = {}

    if seed.value_sha256:
        params["value_sha256"] = seed.value_sha256
    if seed.dataset_name:
        params["dataset_name"] = seed.dataset_name
    if seed.name:
        params["name"] = seed.name

    return ComponentIdentifier(
        class_name=seed.__class__.__name__,
        class_module=seed.__class__.__module__,
        params=params,
    )


def build_atomic_attack_identifier(
    *,
    attack_identifier: ComponentIdentifier,
    seed_group: Optional["SeedGroup"] = None,
) -> ComponentIdentifier:
    """
    Build a composite ComponentIdentifier for an atomic attack.

    Combines the attack strategy's identity with identifiers for the general
    technique seeds from the seed group. Only seeds with ``is_general_technique=True``
    contribute to the identity; objectives and other non-technique seeds are excluded.

    When no seed_group is provided, the resulting identifier has an empty
    ``general_technique_seeds`` children list, but still has the standard
    ``AtomicAttack`` shape for consistent querying.

    Args:
        attack_identifier (ComponentIdentifier): The attack strategy's identifier
            (from ``attack.get_identifier()``).
        seed_group (Optional[SeedGroup]): The seed group to extract general technique
            seeds from. If None, the identifier has empty technique seeds.

    Returns:
        ComponentIdentifier: A composite identifier with class_name="AtomicAttack",
            the attack as a child, and general technique seed identifiers as children.
    """
    seed_identifiers: list[ComponentIdentifier] = []

    if seed_group is not None:
        for seed in seed_group.seeds:
            if seed.is_general_technique:
                seed_identifiers.append(build_seed_identifier(seed))

    children: dict[str, Any] = {
        "attack": attack_identifier,
    }
    if seed_identifiers:
        children["general_technique_seeds"] = seed_identifiers

    return ComponentIdentifier(
        class_name=_ATOMIC_ATTACK_CLASS_NAME,
        class_module=_ATOMIC_ATTACK_CLASS_MODULE,
        children=children,
    )


def _build_attack_eval_dict(
    identifier: ComponentIdentifier,
    *,
    param_allowlist: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Build a dictionary for attack eval hashing.

    Recursively projects an atomic attack identifier to its behavioral
    parameters only, filtering out operational params from target children.

    Args:
        identifier (ComponentIdentifier): The component identity to process.
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
            k: v
            for k, v in sorted(identifier.params.items())
            if v is not None and (param_allowlist is None or k in param_allowlist)
        }
    )

    if identifier.children:
        eval_children: dict[str, Any] = {}
        for name in sorted(identifier.children):
            child_list = identifier.get_child_list(name)
            if name in _TARGET_CHILD_KEYS:
                # Targets: filter to behavioral params only
                hashes = [
                    config_hash(_build_attack_eval_dict(c, param_allowlist=_BEHAVIORAL_CHILD_PARAMS))
                    for c in child_list
                ]
            else:
                # Non-targets (attack children, seed children): full eval treatment
                hashes = [config_hash(_build_attack_eval_dict(c)) for c in child_list]
            eval_children[name] = hashes[0] if len(hashes) == 1 else hashes
        if eval_children:
            eval_dict["children"] = eval_children

    return eval_dict


def compute_attack_eval_hash(identifier: ComponentIdentifier) -> str:
    """
    Compute a behavioral equivalence hash for attack evaluation grouping.

    Projects the composite atomic attack identifier to behavioral params only.
    For target children (objective_target, etc.), only model_name, temperature,
    and top_p are included. For seed children, all params are included.

    This enables grouping attack results by behavioral equivalence when
    running evaluations, similar to ``compute_eval_hash`` for scorers.

    Args:
        identifier (ComponentIdentifier): The atomic attack's composite identity.

    Returns:
        str: A hash suitable for evaluation registry keying.
    """
    return config_hash(_build_attack_eval_dict(identifier))
