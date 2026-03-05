# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.identifiers import (
    AttackEvaluationIdentity,
    ComponentIdentifier,
    build_atomic_attack_identifier,
    build_seed_identifier,
    compute_attack_eval_hash,
    compute_eval_hash,
)
from pyrit.models.seeds.seed_prompt import SeedPrompt


class _FakeSeedGroup:
    """Minimal stub for SeedGroup with a seeds list."""

    def __init__(self, *, seeds: list):
        self.seeds = seeds


class TestBuildSeedIdentifier:
    """Tests for build_seed_identifier."""

    def test_returns_component_identifier(self):
        seed = SeedPrompt(value="hello", value_sha256="abc123", dataset_name="test_ds", name="seed1")
        result = build_seed_identifier(seed)
        assert isinstance(result, ComponentIdentifier)

    def test_captures_class_name(self):
        seed = SeedPrompt(value="hello", value_sha256="abc123")
        result = build_seed_identifier(seed)
        assert result.class_name == "SeedPrompt"

    def test_captures_seed_prompt_class(self):
        seed = SeedPrompt(value="hello", value_sha256="abc123")
        result = build_seed_identifier(seed)
        assert result.class_name == "SeedPrompt"

    def test_includes_value_sha256(self):
        seed = SeedPrompt(value="hello", value_sha256="abc123")
        result = build_seed_identifier(seed)
        assert result.params["value_sha256"] == "abc123"

    def test_includes_dataset_name(self):
        seed = SeedPrompt(value="hello", value_sha256="abc", dataset_name="my_dataset")
        result = build_seed_identifier(seed)
        assert result.params["dataset_name"] == "my_dataset"

    def test_includes_name(self):
        seed = SeedPrompt(value="hello", value_sha256="abc", name="prompt_name")
        result = build_seed_identifier(seed)
        assert result.params["name"] == "prompt_name"

    def test_excludes_none_values(self):
        seed = SeedPrompt(value="hello")
        seed.value_sha256 = None
        seed.dataset_name = None
        seed.name = None
        result = build_seed_identifier(seed)
        assert "value_sha256" not in result.params
        assert "dataset_name" not in result.params
        assert "name" not in result.params

    def test_deterministic_hash(self):
        seed1 = SeedPrompt(value="hello", value_sha256="abc123", dataset_name="ds")
        seed2 = SeedPrompt(value="hello", value_sha256="abc123", dataset_name="ds")
        assert build_seed_identifier(seed1).hash == build_seed_identifier(seed2).hash

    def test_different_content_different_hash(self):
        seed1 = SeedPrompt(value="hello", value_sha256="abc123")
        seed2 = SeedPrompt(value="world", value_sha256="def456")
        assert build_seed_identifier(seed1).hash != build_seed_identifier(seed2).hash


class TestBuildAtomicAttackIdentifier:
    """Tests for build_atomic_attack_identifier."""

    def _make_attack_id(self, *, class_name: str = "PromptSendingAttack") -> ComponentIdentifier:
        return ComponentIdentifier(
            class_name=class_name,
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
        )

    def test_returns_component_identifier(self):
        attack_id = self._make_attack_id()
        result = build_atomic_attack_identifier(attack_identifier=attack_id)
        assert isinstance(result, ComponentIdentifier)

    def test_class_name_is_atomic_attack(self):
        attack_id = self._make_attack_id()
        result = build_atomic_attack_identifier(attack_identifier=attack_id)
        assert result.class_name == "AtomicAttack"

    def test_class_module_is_correct(self):
        attack_id = self._make_attack_id()
        result = build_atomic_attack_identifier(attack_identifier=attack_id)
        assert result.class_module == "pyrit.scenario.core.atomic_attack"

    def test_attack_child_is_present(self):
        attack_id = self._make_attack_id()
        result = build_atomic_attack_identifier(attack_identifier=attack_id)
        assert "attack" in result.children
        assert result.children["attack"] == attack_id

    def test_no_seed_group_no_general_technique_seeds(self):
        attack_id = self._make_attack_id()
        result = build_atomic_attack_identifier(attack_identifier=attack_id)
        assert "general_technique_seeds" not in result.children

    def test_empty_seed_group_no_general_technique_seeds(self):
        attack_id = self._make_attack_id()
        seed_group = _FakeSeedGroup(seeds=[])
        result = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=seed_group)
        assert "general_technique_seeds" not in result.children

    def test_filters_to_general_technique_seeds_only(self):
        attack_id = self._make_attack_id()
        general_seed = SeedPrompt(value="technique", value_sha256="abc", is_general_technique=True)
        non_general_seed = SeedPrompt(value="objective", value_sha256="def", is_general_technique=False)
        seed_group = _FakeSeedGroup(seeds=[general_seed, non_general_seed])
        result = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=seed_group)
        assert "general_technique_seeds" in result.children
        seed_ids = result.children["general_technique_seeds"]
        assert isinstance(seed_ids, list)
        assert len(seed_ids) == 1
        assert seed_ids[0].params.get("value_sha256") == "abc"

    def test_multiple_general_technique_seeds(self):
        attack_id = self._make_attack_id()
        seed1 = SeedPrompt(value="tech1", value_sha256="aaa", is_general_technique=True)
        seed2 = SeedPrompt(value="tech2", value_sha256="bbb", is_general_technique=True)
        seed_group = _FakeSeedGroup(seeds=[seed1, seed2])
        result = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=seed_group)
        seed_ids = result.children["general_technique_seeds"]
        assert len(seed_ids) == 2

    def test_deterministic_hash(self):
        attack_id = self._make_attack_id()
        seed = SeedPrompt(value="technique", value_sha256="abc", is_general_technique=True)
        sg1 = _FakeSeedGroup(seeds=[seed])
        sg2 = _FakeSeedGroup(seeds=[seed])
        r1 = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=sg1)
        r2 = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=sg2)
        assert r1.hash == r2.hash

    def test_different_seeds_different_hash(self):
        attack_id = self._make_attack_id()
        seed1 = SeedPrompt(value="tech1", value_sha256="aaa", is_general_technique=True)
        seed2 = SeedPrompt(value="tech2", value_sha256="bbb", is_general_technique=True)
        r1 = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=_FakeSeedGroup(seeds=[seed1]))
        r2 = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=_FakeSeedGroup(seeds=[seed2]))
        assert r1.hash != r2.hash

    def test_different_attacks_different_hash(self):
        attack_id1 = self._make_attack_id(class_name="PromptSendingAttack")
        attack_id2 = self._make_attack_id(class_name="CrescendoAttack")
        r1 = build_atomic_attack_identifier(attack_identifier=attack_id1)
        r2 = build_atomic_attack_identifier(attack_identifier=attack_id2)
        assert r1.hash != r2.hash

    def test_serialization_round_trip(self):
        attack_id = self._make_attack_id()
        seed = SeedPrompt(value="technique", value_sha256="abc", is_general_technique=True, dataset_name="ds")
        seed_group = _FakeSeedGroup(seeds=[seed])
        original = build_atomic_attack_identifier(attack_identifier=attack_id, seed_group=seed_group)

        serialized = original.to_dict()
        restored = ComponentIdentifier.from_dict(serialized)

        assert restored.class_name == original.class_name
        assert restored.class_module == original.class_module
        assert restored.hash == original.hash


class TestComputeAttackEvalHash:
    """Tests for compute_attack_eval_hash."""

    def _make_attack_id(self) -> ComponentIdentifier:
        return ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
            children={
                "objective_target": ComponentIdentifier(
                    class_name="OpenAIChatTarget",
                    class_module="pyrit.prompt_target.openai.openai_chat_target",
                    params={"model_name": "gpt-4o", "endpoint": "https://api.openai.com", "max_rpm": 100},
                )
            },
        )

    def test_returns_string(self):
        attack_id = self._make_attack_id()
        composite = build_atomic_attack_identifier(attack_identifier=attack_id)
        result = compute_attack_eval_hash(composite)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self):
        attack_id = self._make_attack_id()
        c1 = build_atomic_attack_identifier(attack_identifier=attack_id)
        c2 = build_atomic_attack_identifier(attack_identifier=attack_id)
        assert compute_attack_eval_hash(c1) == compute_attack_eval_hash(c2)

    def test_ignores_operational_params_on_targets(self):
        """Same model, different operational params should produce same eval hash."""
        target1 = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            params={"model_name": "gpt-4o", "endpoint": "https://api1.openai.com", "max_rpm": 100},
        )
        target2 = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            params={"model_name": "gpt-4o", "endpoint": "https://api2.openai.com", "max_rpm": 200},
        )
        attack1 = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
            children={"objective_target": target1},
        )
        attack2 = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
            children={"objective_target": target2},
        )
        c1 = build_atomic_attack_identifier(attack_identifier=attack1)
        c2 = build_atomic_attack_identifier(attack_identifier=attack2)
        assert compute_attack_eval_hash(c1) == compute_attack_eval_hash(c2)

    def test_different_models_different_hash(self):
        """Different model names should produce different eval hash."""
        target1 = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            params={"model_name": "gpt-4o"},
        )
        target2 = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            params={"model_name": "gpt-3.5-turbo"},
        )
        attack1 = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
            children={"objective_target": target1},
        )
        attack2 = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
            children={"objective_target": target2},
        )
        c1 = build_atomic_attack_identifier(attack_identifier=attack1)
        c2 = build_atomic_attack_identifier(attack_identifier=attack2)
        assert compute_attack_eval_hash(c1) != compute_attack_eval_hash(c2)

    def test_different_seeds_different_eval_hash(self):
        attack_id = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
        )
        seed1 = SeedPrompt(value="tech1", value_sha256="aaa", is_general_technique=True)
        seed2 = SeedPrompt(value="tech2", value_sha256="bbb", is_general_technique=True)
        c1 = build_atomic_attack_identifier(
            attack_identifier=attack_id, seed_group=_FakeSeedGroup(seeds=[seed1])
        )
        c2 = build_atomic_attack_identifier(
            attack_identifier=attack_id, seed_group=_FakeSeedGroup(seeds=[seed2])
        )
        assert compute_attack_eval_hash(c1) != compute_attack_eval_hash(c2)


class TestAttackEvaluationIdentity:
    """Tests for AttackEvaluationIdentity."""

    def _make_composite(self) -> ComponentIdentifier:
        attack_id = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module="pyrit.executor.attack.single_turn.prompt_sending",
            children={
                "objective_target": ComponentIdentifier(
                    class_name="OpenAIChatTarget",
                    class_module="pyrit.prompt_target.openai.openai_chat_target",
                    params={"model_name": "gpt-4o", "endpoint": "https://api.openai.com"},
                )
            },
        )
        return build_atomic_attack_identifier(attack_identifier=attack_id)

    def test_identifier_property_returns_original(self):
        composite = self._make_composite()
        identity = AttackEvaluationIdentity(composite)
        assert identity.identifier is composite

    def test_eval_hash_is_64_char_hex(self):
        composite = self._make_composite()
        identity = AttackEvaluationIdentity(composite)
        assert isinstance(identity.eval_hash, str)
        assert len(identity.eval_hash) == 64

    def test_eval_hash_matches_free_function(self):
        """AttackEvaluationIdentity.eval_hash must equal compute_attack_eval_hash."""
        composite = self._make_composite()
        identity = AttackEvaluationIdentity(composite)
        assert identity.eval_hash == compute_attack_eval_hash(composite)

    def test_eval_hash_matches_centralized_compute_eval_hash(self):
        """AttackEvaluationIdentity.eval_hash must equal centralized compute_eval_hash with attack constants."""
        composite = self._make_composite()
        identity = AttackEvaluationIdentity(composite)
        expected = compute_eval_hash(
            composite,
            target_child_keys=AttackEvaluationIdentity.TARGET_CHILD_KEYS,
            behavioral_child_params=AttackEvaluationIdentity.BEHAVIORAL_CHILD_PARAMS,
        )
        assert identity.eval_hash == expected

    def test_target_child_keys_include_objective_target(self):
        """Attack identity filters objective_target only (not prompt_target which is scorer-only)."""
        assert "objective_target" in AttackEvaluationIdentity.TARGET_CHILD_KEYS
        assert "prompt_target" not in AttackEvaluationIdentity.TARGET_CHILD_KEYS
        assert "converter_target" not in AttackEvaluationIdentity.TARGET_CHILD_KEYS

    def test_operational_params_ignored(self):
        """Same model on different endpoints produces same eval hash."""
        target1 = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            params={"model_name": "gpt-4o", "endpoint": "https://api1.openai.com"},
        )
        target2 = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target.openai.openai_chat_target",
            params={"model_name": "gpt-4o", "endpoint": "https://api2.openai.com"},
        )
        attack1 = ComponentIdentifier(
            class_name="Attack", class_module="m", children={"objective_target": target1}
        )
        attack2 = ComponentIdentifier(
            class_name="Attack", class_module="m", children={"objective_target": target2}
        )
        c1 = build_atomic_attack_identifier(attack_identifier=attack1)
        c2 = build_atomic_attack_identifier(attack_identifier=attack2)
        assert AttackEvaluationIdentity(c1).eval_hash == AttackEvaluationIdentity(c2).eval_hash
