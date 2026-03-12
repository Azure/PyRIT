# pyrit.identifiers

Identifiers module for PyRIT components.

## Functions

### build_atomic_attack_identifier

```python
build_atomic_attack_identifier(attack_identifier: ComponentIdentifier, seed_group: Optional[SeedGroup] = None) → ComponentIdentifier
```

Build a composite ComponentIdentifier for an atomic attack.

Combines the attack strategy's identity with identifiers for all seeds
from the seed group. Every seed in the group is included in the identity;
each seed's ``is_general_technique`` flag is captured as a param so that
downstream consumers (e.g., evaluation identity) can filter as needed.

When no seed_group is provided, the resulting identifier has an empty
``seeds`` children list, but still has the standard ``AtomicAttack``
shape for consistent querying.

| Parameter | Type | Description |
|---|---|---|
| `attack_identifier` | `ComponentIdentifier` | The attack strategy's identifier (from ``attack.get_identifier()``). |
| `seed_group` | `Optional[SeedGroup]` | The seed group to extract seeds from. If None, the identifier has an empty seeds list. Defaults to `None`. |

**Returns:**

- `ComponentIdentifier` — A composite identifier with class_name="AtomicAttack",
the attack as a child, and seed identifiers as children.

### `build_seed_identifier(seed: Seed) → ComponentIdentifier`

Build a ComponentIdentifier from a seed's behavioral properties.

Captures the seed's content hash, dataset name, and class type so that
different seeds produce different identifiers while the same seed content
always produces the same identifier.

| Parameter | Type | Description |
|---|---|---|
| `seed` | `Seed` | The seed to build an identifier for. |

**Returns:**

- `ComponentIdentifier` — An identifier capturing the seed's behavioral properties.

### `class_name_to_snake_case(class_name: str, suffix: str = '') → str`

Convert a PascalCase class name to snake_case, optionally stripping a suffix.

| Parameter | Type | Description |
|---|---|---|
| `class_name` | `str` | The class name to convert (e.g., "SelfAskRefusalScorer"). |
| `suffix` | `str` | Optional explicit suffix to strip before conversion (e.g., "Scorer"). Defaults to `''`. |

**Returns:**

- `str` — The snake_case name (e.g., "self_ask_refusal" if suffix="Scorer").

### compute_eval_hash

```python
compute_eval_hash(identifier: ComponentIdentifier, child_eval_rules: dict[str, ChildEvalRule]) → str
```

Compute a behavioral equivalence hash for evaluation grouping.

Unlike ``ComponentIdentifier.hash`` (which includes all params of self and
children), the eval hash applies per-child rules to strip operational params
(like endpoint, max_requests_per_minute), exclude children entirely, or
filter list items.  This ensures the same logical configuration on different
deployments produces the same eval hash.

Children not listed in ``child_eval_rules`` receive full recursive treatment.

When ``child_eval_rules`` is empty, no filtering occurs and the result
equals ``identifier.hash``.

| Parameter | Type | Description |
|---|---|---|
| `identifier` | `ComponentIdentifier` | The component identity to compute the hash for. |
| `child_eval_rules` | `dict[str, ChildEvalRule]` | Per-child eval rules. |

**Returns:**

- `str` — A hex-encoded SHA256 hash suitable for eval registry keying.

### `config_hash(config_dict: dict[str, Any]) → str`

Compute a deterministic SHA256 hash from a config dictionary.

This is the single source of truth for identity hashing across the entire
system. The dict is serialized with sorted keys and compact separators to
ensure determinism.

| Parameter | Type | Description |
|---|---|---|
| `config_dict` | `Dict[str, Any]` | A JSON-serializable dictionary. |

**Returns:**

- `str` — Hex-encoded SHA256 hash string.

**Raises:**

- `TypeError` — If config_dict contains values that are not JSON-serializable.

### `snake_case_to_class_name(snake_case_name: str, suffix: str = '') → str`

Convert a snake_case name to a PascalCase class name.

| Parameter | Type | Description |
|---|---|---|
| `snake_case_name` | `str` | The snake_case name to convert (e.g., "my_custom"). |
| `suffix` | `str` | Optional suffix to append to the class name (e.g., "Scenario" would convert "my_custom" to "MyCustomScenario"). Defaults to `''`. |

**Returns:**

- `str` — The PascalCase class name (e.g., "MyCustomScenario").

## `class AtomicAttackEvaluationIdentifier(EvaluationIdentifier)`

Evaluation identity for atomic attacks.

Per-child rules:

* ``objective_target`` — include only ``temperature``.
* ``adversarial_chat`` — include ``model_name``, ``temperature``, ``top_p``.
* ``objective_scorer`` — excluded entirely.
* ``seeds`` — include only items where ``is_general_technique=True``.

Non-target children (e.g., ``request_converters``, ``response_converters``)
receive full recursive eval treatment, meaning they fully contribute to
the hash.

## `class ChildEvalRule`

Per-child configuration for eval-hash computation.

Controls how a specific named child is treated when building the
evaluation hash:

* ``exclude`` — if ``True``, drop this child entirely from the hash.
* ``included_params`` — if set, only include these param keys for this
  child (and its recursive descendants). ``None`` means all params.
* ``included_item_values`` — for list-valued children, only include items
  whose ``params`` match **all** specified key-value pairs. ``None``
  means include all items.

## `class ComponentIdentifier`

Immutable snapshot of a component's behavioral configuration.

A single type for all component identity — scorers, targets, converters, and
any future component types all produce a ComponentIdentifier with their relevant
params and children.

The hash is content-addressed: two ComponentIdentifiers with the same class, params,
and children produce the same hash. This enables deterministic metrics lookup,
DB deduplication, and registry keying.

**Methods:**

#### `from_dict(data: dict[str, Any]) → ComponentIdentifier`

Deserialize from a stored dictionary.

Reconstructs a ComponentIdentifier from data previously saved via to_dict().
Handles both the current format (``class_name``/``class_module``) and legacy
format (``__type__``/``__module__``) for backward compatibility with
older database records.

| Parameter | Type | Description |
|---|---|---|
| `data` | `Dict[str, Any]` | Dictionary from DB/JSONL storage. The original dict is not mutated; a copy is made internally. |

**Returns:**

- `ComponentIdentifier` — Reconstructed identifier with the stored hash
preserved (if available) to maintain correct identity despite
potential param truncation.

#### `get_child(key: str) → Optional[ComponentIdentifier]`

Get a single child by key.

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | The child key. |

**Returns:**

- `Optional[ComponentIdentifier]` — Optional[ComponentIdentifier]: The child, or None if not found.

**Raises:**

- `ValueError` — If the child is a list (use get_child_list instead).

#### `get_child_list(key: str) → list[ComponentIdentifier]`

Get a list of children by key.

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | The child key. |

**Returns:**

- `list[ComponentIdentifier]` — List[ComponentIdentifier]: The children. Returns empty list if
not found, wraps single child in a list.

#### normalize

```python
normalize(value: Union[ComponentIdentifier, dict[str, Any]]) → ComponentIdentifier
```

Normalize a value to a ComponentIdentifier instance.

Accepts either an existing ComponentIdentifier (returned as-is) or a dict
(reconstructed via from_dict). This supports code paths that may receive
either typed identifiers or raw dicts from database storage.

| Parameter | Type | Description |
|---|---|---|
| `value` | `Union[ComponentIdentifier, Dict[str, Any]]` | A ComponentIdentifier or a dictionary representation. |

**Returns:**

- `ComponentIdentifier` — The normalized identifier instance.

**Raises:**

- `TypeError` — If value is neither a ComponentIdentifier nor a dict.

#### of

```python
of(obj: object, params: Optional[dict[str, Any]] = None, children: Optional[dict[str, Union[ComponentIdentifier, list[ComponentIdentifier]]]] = None) → ComponentIdentifier
```

Build a ComponentIdentifier from a live object instance.

This factory method extracts class_name and class_module from the object's
type automatically, making it the preferred way to create identifiers in
component implementations. None-valued params and children are filtered out
to ensure backward-compatible hashing.

| Parameter | Type | Description |
|---|---|---|
| `obj` | `object` | The live component instance whose type info will be captured. |
| `params` | `Optional[Dict[str, Any]]` | Behavioral parameters that affect the component's output. Only include params that change behavior — exclude operational settings like rate limits, retry counts, or logging config. Defaults to `None`. |
| `children` | `Optional[Dict[str, Union[ComponentIdentifier, List[ComponentIdentifier]]]]` |  Named child component identifiers. Use for compositional components like scorers that wrap other scorers or targets that chain converters. Defaults to `None`. |

**Returns:**

- `ComponentIdentifier` — The frozen identity snapshot with computed hash.

#### `to_dict(max_value_length: Optional[int] = None) → dict[str, Any]`

Serialize to a JSON-compatible dictionary for DB/JSONL storage.

Produces a flat structure where params are inlined at the top level alongside
class_name, class_module, hash, and pyrit_version.

Children are recursively serialized into a nested "children" key.

| Parameter | Type | Description |
|---|---|---|
| `max_value_length` | `Optional[int]` | If provided, string param values longer than this limit are truncated and suffixed with "...". Useful for DB storage where column sizes may be limited. The truncation applies only to param values, not to structural keys like class_name or hash. The limit is propagated to children. Defaults to None (no truncation). Defaults to `None`. |

**Returns:**

- `dict[str, Any]` — Dict[str, Any]: JSON-serializable dictionary suitable for database storage
or JSONL export.

## `class EvaluationIdentifier(ABC)`

Wraps a ``ComponentIdentifier`` with domain-specific eval-hash configuration.

Subclasses set ``CHILD_EVAL_RULES`` — a mapping of child names to
``ChildEvalRule`` instances that control how each child is treated during
eval-hash computation.  Children not listed receive full recursive treatment.

The concrete ``eval_hash`` property delegates to the module-level
``compute_eval_hash`` free function.

## `class Identifiable(ABC)`

Abstract base class for components that provide a behavioral identity.

Components implement ``_build_identifier()`` to return a frozen ComponentIdentifier
snapshot. The identifier is built lazily on first access and cached for the
component's lifetime.

**Methods:**

#### `get_identifier() → ComponentIdentifier`

Get the component's identifier, building it lazily on first access.

The identifier is computed once via _build_identifier() and then cached for
subsequent calls. This ensures consistent identity throughout the
component's lifetime while deferring computation until actually needed.

**Returns:**

- `ComponentIdentifier` — The frozen identity snapshot representing
this component's behavioral configuration.

## `class ScorerEvaluationIdentifier(EvaluationIdentifier)`

Evaluation identity for scorers.

The ``prompt_target`` child is filtered to behavioral params only
(``model_name``, ``temperature``, ``top_p``), so the same scorer
configuration on different deployments produces the same eval hash.
