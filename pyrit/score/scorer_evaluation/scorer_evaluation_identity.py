# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scorer-specific evaluation identity.

``ScorerEvaluationIdentity`` declares which children are "targets" and which
target params are behavioral for the scorer evaluation domain.
"""

from __future__ import annotations

from typing import ClassVar

from pyrit.identifiers.evaluation_identity import EvaluationIdentity


class ScorerEvaluationIdentity(EvaluationIdentity):
    """
    Evaluation identity for scorers.

    Target children (``prompt_target``, ``converter_target``) are filtered to
    behavioral params only (``model_name``, ``temperature``, ``top_p``), so the
    same scorer configuration on different deployments produces the same eval hash.
    """

    TARGET_CHILD_KEYS: ClassVar[frozenset[str]] = frozenset({"prompt_target", "converter_target"})
    BEHAVIORAL_CHILD_PARAMS: ClassVar[frozenset[str]] = frozenset({"model_name", "temperature", "top_p"})
