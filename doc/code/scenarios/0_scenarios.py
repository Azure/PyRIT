# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# Scenarios

# A `Scenario` is a higher-level construct that groups multiple Attack Configurations together. This allows you to execute a comprehensive testing campaign with multiple attack methods sequentially. Scenarios are meant to be configured and written to test for specific workflows. As such, it is okay to hard code some values.

## What is a Scenario?

# A `Scenario` represents a comprehensive testing campaign composed of multiple atomic attack tests. It orchestrates the execution of multiple `AtomicAttack` instances sequentially and aggregates the results into a single `ScenarioResult`.

### Key Components

# - **Scenario**: The top-level orchestrator that groups and executes multiple atomic attacks
# - **AtomicAttack**: An atomic test unit combining an attack strategy, objectives, and execution parameters
# - **ScenarioResult**: Contains the aggregated results from all atomic attacks and scenario metadata

## Use Cases

# Some examples of scenarios you might create:

# - **VibeCheckScenario**: Randomly selects a few prompts from HarmBench to quickly assess model behavior
# - **QuickViolence**: Checks how resilient a model is to violent objectives using multiple attack techniques
# - **ComprehensiveFoundry**: Tests a target with all available attack converters and strategies
# - **CustomCompliance**: Tests against specific compliance requirements with curated datasets and attacks

# These Scenarios can be updated and added to as you refine what you are testing for.

## How It Works

# Each `Scenario` contains a collection of `AtomicAttack` objects. When executed:

# 1. Each `AtomicAttack` is executed sequentially
# 2. Every `AtomicAttack` tests its configured attack against all specified objectives and datasets
# 3. Results are aggregated into a single `ScenarioResult` with all attack outcomes
# 4. Optional memory labels help track and categorize the scenario execution

## Creating Custom Scenarios

# To create a custom scenario, extend the `Scenario` base class and implement the required abstract methods.

### Required Components

# 1. **Strategy Enum**: Create a `ScenarioStrategy` enum that defines the available strategies for your scenario.
#    - Each enum member is defined as `(value, tags)` where value is a string and tags is a set of strings
#    - Include an `ALL` aggregate strategy that expands to all available strategies
#    - Optionally implement `supports_composition()` and `validate_composition()` for strategy composition rules

# 2. **Scenario Class**: Extend `Scenario` and implement these abstract methods:
#    - `get_strategy_class()`: Return your strategy enum class
#    - `get_default_strategy()`: Return the default strategy (typically `YourStrategy.ALL`)
#    - `_get_atomic_attacks_async()`: Build and return a list of `AtomicAttack` instances

# 3. **Constructor**: Use `@apply_defaults` decorator and call `super().__init__()` with scenario metadata:
#    - `name`: Descriptive name for your scenario
#    - `version`: Integer version number
#    - `objective_target`: The target system being tested
#    - `objective_scorer_identifier`: Identifier for the scoring mechanism
#    - `memory_labels`: Optional labels for tracking
#    - `max_concurrency`: Number of concurrent operations (default: 10)
#    - `max_retries`: Number of retry attempts on failure (default: 0)

### Example Structure
# %%
from typing import List, Optional, Type

from pyrit.common import apply_defaults
from pyrit.executor.attack import AttackScoringConfig, PromptSendingAttack
from pyrit.scenarios import AtomicAttack, Scenario, ScenarioStrategy
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class MyStrategy(ScenarioStrategy):
    ALL = ("all", {"all"})
    StrategyA = ("strategy_a", {"tag1", "tag2"})
    StrategyB = ("strategy_b", {"tag1"})


class MyScenario(Scenario):
    version: int = 1

    @classmethod
    def get_strategy_class(cls) -> Type[ScenarioStrategy]:
        return MyStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        return MyStrategy.ALL

    @apply_defaults
    def __init__(
        self,
        *,
        objective_scorer: Optional[TrueFalseScorer] = None,
    ):

        self._scorer_config = AttackScoringConfig(objective_scorer=objective_scorer)

        # Call parent constructor
        super().__init__(
            name="My Custom Scenario",
            version=self.version,
            strategy_class=MyStrategy,
            default_aggregate=MyStrategy.ALL,
            objective_scorer_identifier=objective_scorer.get_identifier(),
        )

    async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
        atomic_attacks = []
        assert self._objective_target is not None
        for strategy in self._strategy_compositions:  # type: ignore
            # Create attack instances based on strategy
            attack = PromptSendingAttack(
                objective_target=self._objective_target,
                attack_scoring_config=self._scorer_config,
            )
            atomic_attacks.append(
                AtomicAttack(
                    atomic_attack_name=strategy.name,
                    attack=attack,
                    objectives=["objective1", "objective2"],
                    memory_labels=self._memory_labels,
                )
            )
        return atomic_attacks


# %% [markdown]

### Existing Scenarios

# - **EncodingScenario**: Tests encoding attacks (Base64, ROT13, etc.) with seed prompts and decoding templates
# - **FoundryScenario**: Comprehensive converter and multi-turn attack testing with difficulty-based strategies
# - **ContentHarmScenario**: Tests harm categories (hate, violence, etc.) by loading datasets from CentralMemory

# See [`FoundryScenario`](../../../pyrit/scenarios/scenarios/foundry_scenario.py),
# [`EncodingScenario`](../../../pyrit/scenarios/scenarios/encoding_scenario.py), and
# [`ContentHarmScenario`](../../../pyrit/scenarios/scenarios/e2e/content_harm_scenario.py) for complete examples.

## Resiliency

# Scenarios can run for a long time, and because of that, things can go wrong. Network issues, rate limits, or other transient failures can interrupt execution. PyRIT provides built-in resiliency features to handle these situations gracefully.

### Automatic Resume

# If you re-run a `scenario`, it will automatically start where it left off. The framework tracks completed attacks and objectives in memory, so you won't lose progress if something interrupts your scenario execution. This means you can safely stop and restart scenarios without duplicating work.

### Retry Mechanism

# You can utilize the `max_retries` parameter to handle transient failures. If any unknown exception occurs during execution, PyRIT will automatically retry the failed operation (starting where it left off) up to the specified number of times. This helps ensure your scenario completes successfully even in the face of temporary issues.

### Dynamic Configuration

# During a long-running scenario, you may want to adjust parameters like `max_concurrency` to manage resource usage, or switch your scorer to use a different target. PyRIT's resiliency features make it safe to stop, reconfigure, and continue scenarios as needed.

# For more information, see [resiliency](../setup/2_resiliency.ipynb)
