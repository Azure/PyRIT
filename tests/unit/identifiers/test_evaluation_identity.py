# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for pyrit.identifiers.evaluation_identity.

Covers the ``EvaluationIdentity`` abstract base class, the ``_build_eval_dict``
helper, and the ``compute_eval_hash`` free function.
"""

from typing import ClassVar

import pytest

from pyrit.identifiers import ComponentIdentifier, compute_eval_hash
from pyrit.identifiers.evaluation_identity import EvaluationIdentity, _build_eval_dict

# ---------------------------------------------------------------------------
# Concrete subclass for testing the ABC
# ---------------------------------------------------------------------------


class _StubEvaluationIdentity(EvaluationIdentity):
    """Minimal concrete subclass for testing the abstract base class."""

    TARGET_CHILD_KEYS: ClassVar[frozenset[str]] = frozenset({"my_target"})
    BEHAVIORAL_CHILD_PARAMS: ClassVar[frozenset[str]] = frozenset({"model_name"})


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_TARGET_CHILD_KEYS = frozenset({"prompt_target", "converter_target"})
_BEHAVIORAL_CHILD_PARAMS = frozenset({"model_name", "temperature", "top_p"})


class TestBuildEvalDict:
    """Tests for _build_eval_dict filtering logic."""

    def test_target_child_params_filtered(self):
        """Test that target children only keep behavioral params."""
        child = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "endpoint": "https://example.com"},
        )
        identifier = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child},
        )

        result = _build_eval_dict(
            identifier,
            target_child_keys=_TARGET_CHILD_KEYS,
            behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS,
        )

        # "endpoint" must not appear anywhere in the child sub-dict
        assert "endpoint" not in str(result)
        assert "children" in result

    def test_non_target_child_params_kept(self):
        """Test that non-target children keep all params (full recursive treatment)."""
        child = ComponentIdentifier(
            class_name="SubScorer",
            class_module="pyrit.score",
            params={"threshold": 0.5, "extra": "value"},
        )
        identifier = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"sub_scorer": child},
        )

        result = _build_eval_dict(
            identifier,
            target_child_keys=_TARGET_CHILD_KEYS,
            behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS,
        )

        assert "children" in result

    def test_no_children_produces_flat_dict(self):
        """Test that an identifier with no children produces a dict without 'children' key."""
        identifier = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            params={"threshold": 0.5},
        )

        result = _build_eval_dict(
            identifier,
            target_child_keys=_TARGET_CHILD_KEYS,
            behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS,
        )

        assert "children" not in result
        assert result[ComponentIdentifier.KEY_CLASS_NAME] == "Scorer"


class TestComputeEvalHash:
    """Tests for the compute_eval_hash free function."""

    def test_deterministic(self):
        """Test that the same identifier + config produces the same hash."""
        identifier = ComponentIdentifier(class_name="Scorer", class_module="pyrit.score")
        h1 = compute_eval_hash(
            identifier, target_child_keys=_TARGET_CHILD_KEYS, behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS
        )
        h2 = compute_eval_hash(
            identifier, target_child_keys=_TARGET_CHILD_KEYS, behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS
        )
        assert h1 == h2

    def test_empty_target_child_keys_returns_component_hash(self):
        """Test that empty target_child_keys bypasses filtering and returns component hash."""
        child = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "endpoint": "https://example.com"},
        )
        identifier = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child},
        )

        result = compute_eval_hash(
            identifier,
            target_child_keys=frozenset(),
            behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS,
        )
        assert result == identifier.hash

    def test_returns_64_char_hex(self):
        """Test that the hash is a 64-char lowercase hex string (SHA-256)."""
        identifier = ComponentIdentifier(class_name="S", class_module="m")
        result = compute_eval_hash(
            identifier, target_child_keys=_TARGET_CHILD_KEYS, behavioral_child_params=_BEHAVIORAL_CHILD_PARAMS
        )
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestEvaluationIdentity:
    """Tests for the EvaluationIdentity abstract base class."""

    def test_identifier_property_returns_original(self):
        """Test that .identifier returns the ComponentIdentifier passed at construction."""
        cid = ComponentIdentifier(class_name="Scorer", class_module="pyrit.score")
        identity = _StubEvaluationIdentity(cid)
        assert identity.identifier is cid

    def test_eval_hash_is_string(self):
        """Test that .eval_hash is a valid hex string."""
        cid = ComponentIdentifier(class_name="Scorer", class_module="pyrit.score")
        identity = _StubEvaluationIdentity(cid)
        assert isinstance(identity.eval_hash, str)
        assert len(identity.eval_hash) == 64

    def test_eval_hash_matches_free_function(self):
        """Test that .eval_hash matches calling compute_eval_hash directly."""
        cid = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            params={"threshold": 0.5},
        )
        identity = _StubEvaluationIdentity(cid)

        expected = compute_eval_hash(
            cid,
            target_child_keys=_StubEvaluationIdentity.TARGET_CHILD_KEYS,
            behavioral_child_params=_StubEvaluationIdentity.BEHAVIORAL_CHILD_PARAMS,
        )
        assert identity.eval_hash == expected

    def test_eval_hash_differs_from_component_hash_when_target_filtered(self):
        """Test that eval hash differs from component hash when target children have operational params."""
        child = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "endpoint": "https://example.com"},
        )
        cid = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"my_target": child},
        )
        identity = _StubEvaluationIdentity(cid)

        # "endpoint" is operational, so eval hash should differ from full component hash
        assert identity.eval_hash != cid.hash

    def test_cannot_instantiate_abc_directly(self):
        """Test that EvaluationIdentity cannot be instantiated without ClassVars."""
        with pytest.raises(AttributeError):
            EvaluationIdentity(ComponentIdentifier(class_name="X", class_module="m"))  # type: ignore[abstract]

    def test_custom_classvars_produce_expected_hash(self):
        """Test that a concrete subclass with custom ClassVars produces the correct eval hash."""

        class CustomIdentity(EvaluationIdentity):
            TARGET_CHILD_KEYS: ClassVar[frozenset[str]] = frozenset({"special_target"})
            BEHAVIORAL_CHILD_PARAMS: ClassVar[frozenset[str]] = frozenset({"model_name", "temperature"})

        child = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "temperature": 0.7, "endpoint": "https://example.com"},
        )
        cid = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"special_target": child},
        )
        identity = CustomIdentity(cid)

        expected = compute_eval_hash(
            cid,
            target_child_keys=frozenset({"special_target"}),
            behavioral_child_params=frozenset({"model_name", "temperature"}),
        )
        assert identity.eval_hash == expected
