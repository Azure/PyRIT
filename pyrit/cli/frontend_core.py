# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared core logic for PyRIT Frontends.

This module contains all the business logic for:
- Loading and discovering scenarios
- Running scenarios
- Formatting output
- Managing initialization scripts

Both pyrit_scan and pyrit_shell use these functions.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypedDict

try:
    import termcolor  # type: ignore

    HAS_TERMCOLOR = True
except ImportError:
    HAS_TERMCOLOR = False

    # Create a dummy termcolor module for fallback
    class termcolor:  # type: ignore
        """Dummy termcolor fallback for colored printing if termcolor is not installed."""

        @staticmethod
        def cprint(text: str, color: str = None, attrs: list = None) -> None:  # type: ignore
            """Print text without color."""
            print(text)


if TYPE_CHECKING:
    from pyrit.cli.initializer_registry import InitializerInfo, InitializerRegistry
    from pyrit.cli.scenario_registry import ScenarioRegistry
    from pyrit.models.scenario_result import ScenarioResult

logger = logging.getLogger(__name__)

# Database type constants
IN_MEMORY = "InMemory"
SQLITE = "SQLite"
AZURE_SQL = "AzureSQL"


class ScenarioInfo(TypedDict):
    """Type definition for scenario information dictionary."""

    name: str
    class_name: str
    description: str
    default_strategy: str
    all_strategies: list[str]
    aggregate_strategies: list[str]
    required_datasets: list[str]


class FrontendCore:
    """
    Shared context for PyRIT operations.

    This object holds all the registries and configuration needed to run
    scenarios. It can be created once (for shell) or per-command (for CLI).
    """

    def __init__(
        self,
        *,
        database: str = SQLITE,
        initialization_scripts: Optional[list[Path]] = None,
        initializer_names: Optional[list[str]] = None,
        log_level: str = "WARNING",
    ):
        """
        Initialize PyRIT context.

        Args:
            database: Database type (InMemory, SQLite, or AzureSQL).
            initialization_scripts: Optional list of initialization script paths.
            initializer_names: Optional list of built-in initializer names to run.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to WARNING.

        Raises:
            ValueError: If database or log_level are invalid.
        """
        # Validate inputs
        self._database = validate_database(database=database)
        self._initialization_scripts = initialization_scripts
        self._initializer_names = initializer_names
        self._log_level = validate_log_level(log_level=log_level)

        # Lazy-loaded registries
        self._scenario_registry: Optional[ScenarioRegistry] = None
        self._initializer_registry: Optional[InitializerRegistry] = None
        self._initialized = False

        # Configure logging
        logging.basicConfig(level=getattr(logging, self._log_level))

    async def initialize_async(self) -> None:
        """Initialize PyRIT and load registries (heavy operation)."""
        if self._initialized:
            return

        from pyrit.cli.initializer_registry import InitializerRegistry
        from pyrit.cli.scenario_registry import ScenarioRegistry
        from pyrit.setup import initialize_pyrit_async

        # Initialize PyRIT without initializers (they run per-scenario)
        await initialize_pyrit_async(
            memory_db_type=self._database,
            initialization_scripts=None,
            initializers=None,
        )

        # Load registries
        self._scenario_registry = ScenarioRegistry()
        if self._initialization_scripts:
            print("Discovering user scenarios...")
            sys.stdout.flush()
            self._scenario_registry.discover_user_scenarios()

        self._initializer_registry = InitializerRegistry()

        self._initialized = True

    @property
    def scenario_registry(self) -> "ScenarioRegistry":
        """
        Get the scenario registry. Must call await initialize_async() first.

        Raises:
            RuntimeError: If initialize_async() has not been called.
        """
        if not self._initialized:
            raise RuntimeError(
                "FrontendCore not initialized. Call 'await context.initialize_async()' before accessing registries."
            )
        assert self._scenario_registry is not None
        return self._scenario_registry

    @property
    def initializer_registry(self) -> "InitializerRegistry":
        """
        Get the initializer registry. Must call await initialize_async() first.

        Raises:
            RuntimeError: If initialize_async() has not been called.
        """
        if not self._initialized:
            raise RuntimeError(
                "FrontendCore not initialized. Call 'await context.initialize_async()' before accessing registries."
            )
        assert self._initializer_registry is not None
        return self._initializer_registry


async def list_scenarios_async(*, context: FrontendCore) -> list[ScenarioInfo]:
    """
    List all available scenarios.

    Args:
        context: PyRIT context with loaded registries.

    Returns:
        List of scenario info dictionaries.
    """
    if not context._initialized:
        await context.initialize_async()
    return context.scenario_registry.list_scenarios()


async def list_initializers_async(
    *, context: FrontendCore, discovery_path: Optional[Path] = None
) -> "Sequence[InitializerInfo]":
    """
    List all available initializers.

    Args:
        context: PyRIT context with loaded registries.
        discovery_path: Optional path to discover initializers from.

    Returns:
        Sequence of initializer info dictionaries.
    """
    if discovery_path:
        from pyrit.cli.initializer_registry import InitializerRegistry

        registry = InitializerRegistry(discovery_path=discovery_path)
        return registry.list_initializers()

    if not context._initialized:
        await context.initialize_async()
    return context.initializer_registry.list_initializers()


async def run_scenario_async(
    *,
    scenario_name: str,
    context: FrontendCore,
    scenario_strategies: Optional[list[str]] = None,
    max_concurrency: Optional[int] = None,
    max_retries: Optional[int] = None,
    memory_labels: Optional[dict[str, str]] = None,
    print_summary: bool = True,
) -> "ScenarioResult":
    """
    Run a scenario by name.

    Args:
        scenario_name: Name of the scenario to run.
        context: PyRIT context with loaded registries.
        scenario_strategies: Optional list of strategy names.
        max_concurrency: Max concurrent operations.
        max_retries: Max retry attempts.
        memory_labels: Labels to attach to memory entries.
        print_summary: Whether to print the summary after execution. Defaults to True.

    Returns:
        ScenarioResult: The result of the scenario execution.

    Raises:
        ValueError: If scenario not found or fails to run.

    Note:
        Initializers from PyRITContext will be run before the scenario executes.
    """
    from pyrit.scenario.printer.console_printer import ConsoleScenarioResultPrinter
    from pyrit.setup import initialize_pyrit_async

    # Ensure context is initialized first (loads registries)
    # This must happen BEFORE we run initializers to avoid double-initialization
    if not context._initialized:
        await context.initialize_async()

    # Run initializers before scenario
    initializer_instances = None
    if context._initializer_names:
        print(f"Running {len(context._initializer_names)} initializer(s)...")
        sys.stdout.flush()

        initializer_instances = []

        for name in context._initializer_names:
            initializer_class = context.initializer_registry.get_initializer_class(name=name)
            initializer_instances.append(initializer_class())

    # Re-initialize PyRIT with the scenario-specific initializers
    # This resets memory and applies initializer defaults
    await initialize_pyrit_async(
        memory_db_type=context._database,
        initialization_scripts=context._initialization_scripts,
        initializers=initializer_instances,
    )

    # Get scenario class
    scenario_class = context.scenario_registry.get_scenario(scenario_name)

    if scenario_class is None:
        available = ", ".join(context.scenario_registry.get_scenario_names())
        raise ValueError(f"Scenario '{scenario_name}' not found.\n" f"Available scenarios: {available}")

    # Build initialization kwargs (these go to initialize_async, not __init__)
    init_kwargs: dict[str, Any] = {}

    if scenario_strategies:
        strategy_class = scenario_class.get_strategy_class()
        strategy_enums = []
        for name in scenario_strategies:
            try:
                strategy_enums.append(strategy_class(name))
            except ValueError:
                available_strategies = [s.value for s in strategy_class]
                raise ValueError(
                    f"Strategy '{name}' not found for scenario '{scenario_name}'. "
                    f"Available: {', '.join(available_strategies)}"
                ) from None
        init_kwargs["scenario_strategies"] = strategy_enums

    if max_concurrency is not None:
        init_kwargs["max_concurrency"] = max_concurrency
    if max_retries is not None:
        init_kwargs["max_retries"] = max_retries
    if memory_labels is not None:
        init_kwargs["memory_labels"] = memory_labels

    # Instantiate and run
    print(f"\nRunning scenario: {scenario_name}")
    sys.stdout.flush()

    # Scenarios here are a concrete subclass
    # Runtime parameters are passed to initialize_async()
    scenario = scenario_class()  # type: ignore[call-arg]
    await scenario.initialize_async(**init_kwargs)
    result = await scenario.run_async()

    # Print results if requested
    if print_summary:
        printer = ConsoleScenarioResultPrinter()
        await printer.print_summary_async(result)

    return result


def _format_wrapped_text(*, text: str, indent: str, width: int = 78) -> str:
    """
    Format text with word wrapping.

    Args:
        text: Text to wrap.
        indent: Indentation string for wrapped lines.
        width: Maximum line width. Defaults to 78.

    Returns:
        Formatted text with line breaks.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + len(word) + 1 + len(indent) <= width:
            current_line += " " + word
        else:
            lines.append(indent + current_line)
            current_line = word

    if current_line:
        lines.append(indent + current_line)

    return "\n".join(lines)


def _print_header(*, text: str) -> None:
    """
    Print a colored header if termcolor is available.

    Args:
        text: Header text to print.
    """
    if HAS_TERMCOLOR:
        termcolor.cprint(f"\n  {text}", "cyan", attrs=["bold"])
    else:
        print(f"\n  {text}")


def format_scenario_info(*, scenario_info: ScenarioInfo) -> None:
    """
    Print formatted information about a scenario.

    Args:
        scenario_info: Dictionary containing scenario information.
    """
    _print_header(text=scenario_info["name"])
    print(f"    Class: {scenario_info['class_name']}")

    description = scenario_info.get("description", "")
    if description:
        print("    Description:")
        print(_format_wrapped_text(text=description, indent="      "))

    if scenario_info.get("aggregate_strategies"):
        agg_strategies = scenario_info["aggregate_strategies"]
        print("    Aggregate Strategies:")
        formatted = _format_wrapped_text(text=", ".join(agg_strategies), indent="      - ")
        print(formatted)

    if scenario_info.get("all_strategies"):
        strategies = scenario_info["all_strategies"]
        print(f"    Available Strategies ({len(strategies)}):")
        formatted = _format_wrapped_text(text=", ".join(strategies), indent="      ")
        print(formatted)

    if scenario_info.get("default_strategy"):
        print(f"    Default Strategy: {scenario_info['default_strategy']}")

    if scenario_info.get("required_datasets"):
        datasets = scenario_info["required_datasets"]
        if datasets:
            print(f"    Required Datasets ({len(datasets)}):")
            formatted = _format_wrapped_text(text=", ".join(datasets), indent="      ")
            print(formatted)
        else:
            print("    Required Datasets: None")


def format_initializer_info(*, initializer_info: "InitializerInfo") -> None:
    """
    Print formatted information about an initializer.

    Args:
        initializer_info: Dictionary containing initializer information.
    """
    _print_header(text=initializer_info["name"])
    print(f"    Class: {initializer_info['class_name']}")
    print(f"    Name: {initializer_info['initializer_name']}")
    print(f"    Execution Order: {initializer_info['execution_order']}")

    if initializer_info.get("required_env_vars"):
        print("    Required Environment Variables:")
        for env_var in initializer_info["required_env_vars"]:
            print(f"      - {env_var}")
    else:
        print("    Required Environment Variables: None")

    if initializer_info.get("description"):
        print("    Description:")
        print(_format_wrapped_text(text=initializer_info["description"], indent="      "))


def validate_database(*, database: str) -> str:
    """
    Validate database type.

    Args:
        database: Database type string.

    Returns:
        Validated database type.

    Raises:
        ValueError: If database type is invalid.
    """
    valid_databases = [IN_MEMORY, SQLITE, AZURE_SQL]
    if database not in valid_databases:
        raise ValueError(f"Invalid database type: {database}. " f"Must be one of: {', '.join(valid_databases)}")
    return database


def validate_log_level(*, log_level: str) -> str:
    """
    Validate log level.

    Args:
        log_level: Log level string (case-insensitive).

    Returns:
        Validated log level in uppercase.

    Raises:
        ValueError: If log level is invalid.
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level_upper = log_level.upper()
    if level_upper not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. " f"Must be one of: {', '.join(valid_levels)}")
    return level_upper


def validate_integer(value: str, *, name: str = "value", min_value: Optional[int] = None) -> int:
    """
    Validate and parse an integer value.

    Note: The 'value' parameter is positional (not keyword-only) to allow use with
    argparse lambdas like: lambda v: validate_integer(v, min_value=1).
    This is an exception to the PyRIT style guide for argparse compatibility.

    Args:
        value: String value to parse.
        name: Parameter name for error messages. Defaults to "value".
        min_value: Optional minimum value constraint.

    Returns:
        Parsed integer.

    Raises:
        ValueError: If value is not a valid integer or violates constraints.
    """
    # Reject boolean types explicitly (int(True) == 1, int(False) == 0)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer string, got boolean: {value}")

    # Ensure value is a string
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}: {value}")

    # Strip whitespace and validate it looks like an integer
    value = value.strip()
    if not value:
        raise ValueError(f"{name} cannot be empty")

    try:
        int_value = int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} must be an integer, got: {value}") from e

    if min_value is not None and int_value < min_value:
        raise ValueError(f"{name} must be at least {min_value}, got: {int_value}")

    return int_value


def _argparse_validator(validator_func: Callable[..., Any]) -> Callable[[Any], Any]:
    """
    Adapt a validator to argparse by converting ValueError to ArgumentTypeError.

    This decorator adapts our keyword-only validators for use with argparse's type= parameter.
    It handles two challenges:

    1. Exception Translation: argparse expects ArgumentTypeError, but our validators raise
       ValueError. This decorator catches ValueError and re-raises as ArgumentTypeError.

    2. Keyword-Only Parameters: PyRIT validators use keyword-only parameters (e.g.,
       validate_database(*, database: str)), but argparse's type= passes a positional argument.
       This decorator inspects the function signature and calls the validator with the correct
       keyword argument name.

    This pattern allows us to:
    - Keep validators as pure functions with proper type hints
    - Follow PyRIT style guide (keyword-only parameters)
    - Reuse the same validation logic in both argparse and non-argparse contexts

    Args:
        validator_func: Function that raises ValueError on invalid input.
            Must have at least one parameter (can be keyword-only).

    Returns:
        Wrapped function that:
        - Accepts a single positional argument (for argparse compatibility)
        - Calls validator_func with the correct keyword argument
        - Raises ArgumentTypeError instead of ValueError

    Raises:
        ValueError: If validator_func has no parameters.
    """
    import inspect

    # Get the first parameter name from the function signature
    sig = inspect.signature(validator_func)
    params = list(sig.parameters.keys())
    if not params:
        raise ValueError(f"Validator function {validator_func.__name__} must have at least one parameter")
    first_param = params[0]

    def wrapper(value):
        import argparse as ap

        try:
            # Call with keyword argument to support keyword-only parameters
            return validator_func(**{first_param: value})
        except ValueError as e:
            raise ap.ArgumentTypeError(str(e)) from e

    # Preserve function metadata for better debugging
    wrapper.__name__ = getattr(validator_func, "__name__", "argparse_validator")
    wrapper.__doc__ = getattr(validator_func, "__doc__", None)
    return wrapper


# Argparse-compatible validators
#
# These wrappers adapt our core validators (which use keyword-only parameters and raise
# ValueError) for use with argparse's type= parameter (which passes positional arguments
# and expects ArgumentTypeError).
#
# Pattern:
#   - Use core validators (validate_database, validate_log_level, etc.) in regular code
#   - Use these _argparse versions ONLY in parser.add_argument(..., type=...)
#
# The lambda wrappers for validate_integer are necessary because we need to partially
# apply the min_value parameter while still allowing the decorator to work correctly.
validate_database_argparse = _argparse_validator(validate_database)
validate_log_level_argparse = _argparse_validator(validate_log_level)
positive_int = _argparse_validator(lambda v: validate_integer(v, min_value=1))
non_negative_int = _argparse_validator(lambda v: validate_integer(v, min_value=0))


def parse_memory_labels(json_string: str) -> dict[str, str]:
    """
    Parse memory labels from a JSON string.

    Args:
        json_string: JSON string containing label key-value pairs.

    Returns:
        Dictionary of labels.

    Raises:
        ValueError: If JSON is invalid or contains non-string values.
    """
    try:
        labels = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for memory labels: {e}") from e

    if not isinstance(labels, dict):
        raise ValueError("Memory labels must be a JSON object (dictionary)")

    # Validate all keys and values are strings
    for key, value in labels.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"All label keys and values must be strings. Got: {key}={value}")

    return labels


def resolve_initialization_scripts(script_paths: list[str]) -> list[Path]:
    """
    Resolve initialization script paths.

    Args:
        script_paths: List of script path strings.

    Returns:
        List of resolved Path objects.

    Raises:
        FileNotFoundError: If a script path does not exist.
    """
    from pyrit.cli.initializer_registry import InitializerRegistry

    return InitializerRegistry.resolve_script_paths(script_paths=script_paths)


def get_default_initializer_discovery_path() -> Path:
    """
    Get the default path for discovering initializers.

    Returns:
        Path to the scenarios initializers directory.
    """
    PYRIT_PATH = Path(__file__).parent.parent.resolve()
    return PYRIT_PATH / "setup" / "initializers" / "scenarios"


async def print_scenarios_list_async(*, context: FrontendCore) -> int:
    """
    Print a formatted list of all available scenarios.

    Args:
        context: PyRIT context with loaded registries.

    Returns:
        Exit code (0 for success).
    """
    scenarios = await list_scenarios_async(context=context)

    if not scenarios:
        print("No scenarios found.")
        return 0

    print("\nAvailable Scenarios:")
    print("=" * 80)
    for scenario_info in scenarios:
        format_scenario_info(scenario_info=scenario_info)
    print("\n" + "=" * 80)
    print(f"\nTotal scenarios: {len(scenarios)}")
    return 0


async def print_initializers_list_async(*, context: FrontendCore, discovery_path: Optional[Path] = None) -> int:
    """
    Print a formatted list of all available initializers.

    Args:
        context: PyRIT context with loaded registries.
        discovery_path: Optional path to discover initializers from.

    Returns:
        Exit code (0 for success).
    """
    initializers = await list_initializers_async(context=context, discovery_path=discovery_path)

    if not initializers:
        print("No initializers found.")
        return 0

    print("\nAvailable Initializers:")
    print("=" * 80)
    for initializer_info in initializers:
        format_initializer_info(initializer_info=initializer_info)
    print("\n" + "=" * 80)
    print(f"\nTotal initializers: {len(initializers)}")
    return 0


# Shared argument help text
ARG_HELP = {
    "initializers": "Built-in initializer names to run before the scenario (e.g., openai_objective_target)",
    "initialization_scripts": "Paths to custom Python initialization scripts to run before the scenario",
    "scenario_strategies": "List of strategy names to run (e.g., base64 rot13)",
    "max_concurrency": "Maximum number of concurrent attack executions (must be >= 1)",
    "max_retries": "Maximum number of automatic retries on exception (must be >= 0)",
    "memory_labels": 'Additional labels as JSON string (e.g., \'{"experiment": "test1"}\')',
    "database": "Database type to use for memory storage",
    "log_level": "Logging level",
}


def parse_run_arguments(*, args_string: str) -> dict[str, Any]:
    """
    Parse run command arguments from a string (for shell mode).

    Args:
        args_string: Space-separated argument string (e.g., "scenario_name --initializers foo --strategies bar").

    Returns:
        Dictionary with parsed arguments:
            - scenario_name: str
            - initializers: Optional[list[str]]
            - initialization_scripts: Optional[list[str]]
            - scenario_strategies: Optional[list[str]]
            - max_concurrency: Optional[int]
            - max_retries: Optional[int]
            - memory_labels: Optional[dict[str, str]]
            - database: Optional[str]
            - log_level: Optional[str]

    Raises:
        ValueError: If parsing or validation fails.
    """
    parts = args_string.split()

    if not parts:
        raise ValueError("No scenario name provided")

    result: dict[str, Any] = {
        "scenario_name": parts[0],
        "initializers": None,
        "initialization_scripts": None,
        "scenario_strategies": None,
        "max_concurrency": None,
        "max_retries": None,
        "memory_labels": None,
        "database": None,
        "log_level": None,
    }

    i = 1
    while i < len(parts):
        if parts[i] == "--initializers":
            # Collect initializers until next flag
            result["initializers"] = []
            i += 1
            while i < len(parts) and not parts[i].startswith("--"):
                result["initializers"].append(parts[i])
                i += 1
        elif parts[i] == "--initialization-scripts":
            # Collect script paths until next flag
            result["initialization_scripts"] = []
            i += 1
            while i < len(parts) and not parts[i].startswith("--"):
                result["initialization_scripts"].append(parts[i])
                i += 1
        elif parts[i] in ("--strategies", "-s"):
            # Collect strategies until next flag
            result["scenario_strategies"] = []
            i += 1
            while i < len(parts) and not parts[i].startswith("--") and parts[i] != "-s":
                result["scenario_strategies"].append(parts[i])
                i += 1
        elif parts[i] == "--max-concurrency":
            i += 1
            if i >= len(parts):
                raise ValueError("--max-concurrency requires a value")
            result["max_concurrency"] = validate_integer(parts[i], name="--max-concurrency", min_value=1)
            i += 1
        elif parts[i] == "--max-retries":
            i += 1
            if i >= len(parts):
                raise ValueError("--max-retries requires a value")
            result["max_retries"] = validate_integer(parts[i], name="--max-retries", min_value=0)
            i += 1
        elif parts[i] == "--memory-labels":
            i += 1
            if i >= len(parts):
                raise ValueError("--memory-labels requires a value")
            result["memory_labels"] = parse_memory_labels(parts[i])
            i += 1
        elif parts[i] == "--database":
            i += 1
            if i >= len(parts):
                raise ValueError("--database requires a value")
            result["database"] = validate_database(database=parts[i])
            i += 1
        elif parts[i] == "--log-level":
            i += 1
            if i >= len(parts):
                raise ValueError("--log-level requires a value")
            result["log_level"] = validate_log_level(log_level=parts[i])
            i += 1
        else:
            logger.warning(f"Unknown argument: {parts[i]}")
            i += 1

    return result
