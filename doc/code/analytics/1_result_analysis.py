# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Result Analysis
#
# The `analyze_results` function computes attack success rates from a list of `AttackResult` objects.
# It supports flexible grouping across built-in dimensions (`attack_type`, `converter_type`, `label`)
# as well as composite and custom dimensions.

# %% [markdown]
# ## Setup
#
# First, let's create some sample `AttackResult` objects to work with.

# %%
from pyrit.analytics import analyze_results
from pyrit.identifiers import ConverterIdentifier
from pyrit.models import AttackOutcome, AttackResult, MessagePiece


def make_converter(name: str) -> ConverterIdentifier:
    return ConverterIdentifier(
        class_name=name,
        class_module="pyrit.prompt_converter",
        class_description=f"{name} converter",
        identifier_type="instance",
        supported_input_types=("text",),
        supported_output_types=("text",),
    )


# Realistic attack_identifier dicts mirror Strategy.get_identifier() output
crescendo_id = {
    "__type__": "CrescendoAttack",
    "__module__": "pyrit.executor.attack.multi_turn.crescendo",
    "id": "a1b2c3d4-0001-4000-8000-000000000001",
}
red_team_id = {
    "__type__": "RedTeamingAttack",
    "__module__": "pyrit.executor.attack.multi_turn.red_teaming",
    "id": "a1b2c3d4-0002-4000-8000-000000000002",
}

# Build a small set of representative attack results
results = [
    # Crescendo attacks with Base64Converter
    AttackResult(
        conversation_id="c1",
        objective="bypass safety filter",
        attack_identifier=crescendo_id,
        outcome=AttackOutcome.SUCCESS,
        last_response=MessagePiece(
            role="user",
            original_value="response 1",
            converter_identifiers=[make_converter("Base64Converter")],
            labels={"operation_name": "op_safety_bypass", "operator": "alice"},
        ),
    ),
    AttackResult(
        conversation_id="c2",
        objective="bypass safety filter",
        attack_identifier=crescendo_id,
        outcome=AttackOutcome.FAILURE,
        last_response=MessagePiece(
            role="user",
            original_value="response 2",
            converter_identifiers=[make_converter("Base64Converter")],
            labels={"operation_name": "op_safety_bypass", "operator": "alice"},
        ),
    ),
    # Red teaming attacks with ROT13Converter
    AttackResult(
        conversation_id="c3",
        objective="extract secrets",
        attack_identifier=red_team_id,
        outcome=AttackOutcome.SUCCESS,
        last_response=MessagePiece(
            role="user",
            original_value="response 3",
            converter_identifiers=[make_converter("ROT13Converter")],
            labels={"operation_name": "op_secret_extract", "operator": "bob"},
        ),
    ),
    AttackResult(
        conversation_id="c4",
        objective="extract secrets",
        attack_identifier=red_team_id,
        outcome=AttackOutcome.SUCCESS,
        last_response=MessagePiece(
            role="user",
            original_value="response 4",
            converter_identifiers=[make_converter("ROT13Converter")],
            labels={"operation_name": "op_secret_extract", "operator": "bob"},
        ),
    ),
    # An undetermined result (no converter, no labels)
    AttackResult(
        conversation_id="c5",
        objective="test prompt",
        attack_identifier=crescendo_id,
        outcome=AttackOutcome.UNDETERMINED,
    ),
]

print(f"Created {len(results)} sample AttackResult objects")

# %% [markdown]
# ## Overall Stats (No Grouping)
#
# Pass `group_by=[]` to compute only the overall attack success rate, with no
# dimensional breakdown.

# %%
result = analyze_results(results, group_by=[])

print(f"Overall success rate: {result.overall.success_rate}")
print(f"  Successes:    {result.overall.successes}")
print(f"  Failures:     {result.overall.failures}")
print(f"  Undetermined: {result.overall.undetermined}")
print(f"  Total decided (excl. undetermined): {result.overall.total_decided}")

# %% [markdown]
# ## Group by Attack Type
#
# See how success rates differ across attack strategies (e.g. `crescendo` vs `red_teaming`).

# %%
result = analyze_results(results, group_by=["attack_type"])

for attack_type, stats in result.dimensions["attack_type"].items():
    print(
        f"  {attack_type}: success_rate={stats.success_rate}, "
        f"successes={stats.successes}, failures={stats.failures}, "
        f"undetermined={stats.undetermined}"
    )

# %% [markdown]
# ## Group by Converter Type
#
# Break down success rates by which prompt converter was applied.

# %%
result = analyze_results(results, group_by=["converter_type"])

for converter, stats in result.dimensions["converter_type"].items():
    print(f"  {converter}: success_rate={stats.success_rate}, successes={stats.successes}, failures={stats.failures}")

# %% [markdown]
# ## Group by Label
#
# Labels are key=value metadata attached to messages. Each label pair becomes its own
# grouping key.

# %%
result = analyze_results(results, group_by=["label"])

for label_key, stats in result.dimensions["label"].items():
    print(f"  {label_key}: success_rate={stats.success_rate}, successes={stats.successes}, failures={stats.failures}")

# %% [markdown]
# ## Multiple Dimensions at Once
#
# Pass several dimension names to `group_by` for independent breakdowns in a single call.

# %%
result = analyze_results(results, group_by=["attack_type", "converter_type"])

print("--- By attack_type ---")
for key, stats in result.dimensions["attack_type"].items():
    print(f"  {key}: success_rate={stats.success_rate}")

print("\n--- By converter_type ---")
for key, stats in result.dimensions["converter_type"].items():
    print(f"  {key}: success_rate={stats.success_rate}")

# %% [markdown]
# ## Composite Dimensions
#
# Use a tuple of dimension names to create a cross-product grouping. For example,
# `("converter_type", "attack_type")` produces keys like `("Base64Converter", "crescendo")`.

# %%
result = analyze_results(results, group_by=[("converter_type", "attack_type")])

for combo_key, stats in result.dimensions[("converter_type", "attack_type")].items():
    print(f"  {combo_key}: success_rate={stats.success_rate}, successes={stats.successes}, failures={stats.failures}")

# %% [markdown]
# ## Custom Dimensions
#
# Supply your own extractor function via `custom_dimensions`. An extractor takes an
# `AttackResult` and returns a `list[str]` of dimension values. Here we group by the
# attack objective.

# %%


def extract_objective(attack: AttackResult) -> list[str]:
    return [attack.objective]


result = analyze_results(
    results,
    group_by=["objective"],
    custom_dimensions={"objective": extract_objective},
)

for objective, stats in result.dimensions["objective"].items():
    print(f"  {objective}: success_rate={stats.success_rate}, successes={stats.successes}, failures={stats.failures}")

# %% [markdown]
# ## Default Behavior
#
# When `group_by` is omitted, `analyze_results` groups by **all** registered
# dimensions: `attack_type`, `converter_type`, and `label`.

# %%
result = analyze_results(results)

print(f"Dimensions returned: {list(result.dimensions.keys())}")
print(f"Overall success rate: {result.overall.success_rate}")
