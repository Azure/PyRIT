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

``AttackEvaluationIdentity`` is the attack-domain subclass of
``EvaluationIdentity``, analogous to ``ScorerEvaluationIdentity`` for scorers.
``compute_attack_eval_hash`` is a convenience wrapper around the centralized
``compute_eval_hash`` with attack-specific constants.
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from pyrit.identifiers.component_identifier import ComponentIdentifier
from pyrit.identifiers.evaluation_identity import EvaluationIdentity, compute_eval_hash

if TYPE_CHECKING:
    from pyrit.models.seeds.seed import Seed
    from pyrit.models.seeds.seed_group import SeedGroup

logger = logging.getLogger(__name__)

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


class AttackEvaluationIdentity(EvaluationIdentity):
    """
    Evaluation identity for attacks.

    Target children (``objective_target``, ``prompt_target``, ``converter_target``)
    are filtered to behavioral params only (``model_name``, ``temperature``,
    ``top_p``), so the same attack configuration on different deployments
    produces the same eval hash.  Non-target children (e.g., seed identifiers)
    receive full recursive eval treatment.
    """

    TARGET_CHILD_KEYS: ClassVar[frozenset[str]] = frozenset({"objective_target", "prompt_target", "converter_target"})
    BEHAVIORAL_CHILD_PARAMS: ClassVar[frozenset[str]] = frozenset({"model_name", "temperature", "top_p"})


def compute_attack_eval_hash(identifier: ComponentIdentifier) -> str:
    """
    Compute a behavioral equivalence hash for attack evaluation grouping.

    Convenience wrapper around ``compute_eval_hash`` with attack-specific
    constants.  For target children (objective_target, prompt_target,
    converter_target), only model_name, temperature, and top_p are included.
    For seed children, all params are included.

    This enables grouping attack results by behavioral equivalence when
    running evaluations, similar to ``compute_eval_hash`` for scorers.

    Args:
        identifier (ComponentIdentifier): The atomic attack's composite identity.

    Returns:
        str: A hash suitable for evaluation registry keying.
    """
    return compute_eval_hash(
        identifier,
        target_child_keys=AttackEvaluationIdentity.TARGET_CHILD_KEYS,
        behavioral_child_params=AttackEvaluationIdentity.BEHAVIORAL_CHILD_PARAMS,
    )
