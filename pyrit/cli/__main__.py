# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
PyRIT CLI - Command-line interface for running security scenarios.

This module provides the main entry point for the pyrit_scan command.
It supports running scenarios with configurable database backends and
initialization scripts.
"""

# Standard library imports
import asyncio
import logging
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Any

# Local application imports
from pyrit.cli.initializer_registry import InitializerInfo, InitializerRegistry
from pyrit.cli.scenario_registry import ScenarioRegistry
from pyrit.common.path import PYRIT_PATH
from pyrit.scenarios import Scenario
from pyrit.scenarios.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.setup import AZURE_SQL, IN_MEMORY, SQLITE, initialize_pyrit

logger = logging.getLogger(__name__)


def parse_args(args=None) -> Namespace:
    """
    Parse command-line arguments for the PyRIT scanner.

    Args:
        args: Command-line arguments to parse. If None, uses sys.argv.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(
        prog="pyrit_scan",
        description="""PyRIT Scanner - Run security scenarios against AI systems

Examples:
  # List available scenarios and initializers
  pyrit_scan --list-scenarios
  pyrit_scan --list-initializers

  # Run a scenario with built-in initializers
  pyrit_scan foundry_scenario --initializers simple objective_target

  # Run with custom initialization scripts
  pyrit_scan encoding_scenario --initialization-scripts ./my_config.py

  # Run specific strategies
  pyrit_scan encoding_scenario --initializers simple objective_target --scenario-strategies base64 rot13 morse_code
  pyrit_scan foundry_scenario --initializers simple objective_target --scenario-strategies base64 atbash
""",
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    # Discovery/help options
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available scenarios and exit",
    )

    parser.add_argument(
        "--list-initializers",
        action="store_true",
        help="List all available scenario initializers and exit",
    )

    # Scenario name as positional argument
    parser.add_argument(
        "scenario_name",
        type=str,
        nargs="?",
        help="Name of the scenario to run (e.g., encoding_scenario, foundry_scenario)",
    )

    # Scenario options
    parser.add_argument(
        "--database",
        type=str,
        choices=[IN_MEMORY, SQLITE, AZURE_SQL],
        default=SQLITE,
        help="Database type to use for memory storage",
    )
    parser.add_argument(
        "--initializers",
        type=str,
        nargs="+",
        help="Built-in initializer names (e.g., simple, objective_target, objective_list)",
    )
    parser.add_argument(
        "--initialization-scripts",
        type=str,
        nargs="+",
        help="Paths to custom Python initialization scripts that configure scenarios and defaults",
    )
    parser.add_argument(
        "--scenario-strategies",
        type=str,
        nargs="+",
        help="List of strategy names to run (e.g., base64 rot13 morse_code). If not specified, uses scenario defaults.",
    )

    return parser.parse_args(args)


def list_scenarios(*, registry: ScenarioRegistry) -> None:
    """
    Print all available scenarios to the console.

    Args:
        registry (ScenarioRegistry): The scenario registry to query.
    """
    scenarios = registry.list_scenarios()

    if not scenarios:
        print("No scenarios found.")
        return

    print("\nAvailable Scenarios:")
    print("=" * 80)

    for scenario_info in scenarios:
        _format_scenario_info(scenario_info=scenario_info)

    print("\n" + "=" * 80)
    print(f"\nTotal scenarios: {len(scenarios)}")
    print("\nFor usage information, use: pyrit_scan --help")


def list_initializers() -> None:
    """
    Print all available scenario initializers to the console.

    Discovers initializers from pyrit/setup/initializers/scenarios directory only.
    Note that other initializers (like 'simple' from the base directory) can still
    be used with --initializers, they just won't appear in this listing.
    """
    scenarios_path = Path(PYRIT_PATH) / "setup" / "initializers" / "scenarios"
    registry = InitializerRegistry(discovery_path=scenarios_path)
    initializer_info = registry.list_initializers()

    if not initializer_info:
        print("No initializers found.")
        return

    print("\nAvailable Scenario Initializers:")
    print("=" * 80)

    for info in initializer_info:
        _format_initializer_info(info=info)

    print("\n" + "=" * 80)
    print(f"\nTotal initializers: {len(initializer_info)}")
    print("\nFor usage information, use: pyrit_scan --help")


async def run_scenario_async(
    *,
    scenario_name: str,
    registry: ScenarioRegistry,
    scenario_strategies: list[str] | None = None,
) -> None:
    """
    Run a specific scenario by name.

    Args:
        scenario_name (str): Name of the scenario to run.
        registry (ScenarioRegistry): The scenario registry to query.
        scenario_strategies (list[str] | None): Optional list of strategy names to run.
            If provided, these will be converted to ScenarioCompositeStrategy instances
            and passed to the scenario's __init__.

    Raises:
        ValueError: If the scenario is not found or cannot be instantiated.
    """
    # Get the scenario class
    scenario_class = registry.get_scenario(scenario_name)

    if scenario_class is None:
        available_scenarios = ", ".join(registry.get_scenario_names())
        raise ValueError(
            f"Scenario '{scenario_name}' not found.\n"
            f"Available scenarios: {available_scenarios}\n"
            f"Use 'pyrit_scan --list-scenarios' to see all available scenarios."
        )

    logger.info(f"Instantiating scenario: {scenario_class.__name__}")

    # Instantiate the scenario with optional strategy override
    # The scenario should get its configuration from:
    # 1. --scenario-strategies CLI flag (if provided)
    # 2. Default values set by initialization scripts via @apply_defaults
    # 3. Global variables set by initialization scripts

    scenario: Scenario

    try:
        # Instantiate the scenario (no strategies in __init__)
        scenario = scenario_class()  # type: ignore[call-arg]
    except TypeError as e:
        # Check if this is a missing parameter error
        error_msg = str(e)
        if "missing" in error_msg.lower() and "required" in error_msg.lower():
            raise ValueError(
                f"Failed to instantiate scenario '{scenario_name}'.\n"
                f"The scenario requires parameters that are not configured.\n\n"
                f"To fix this, provide an initialization script with --initialization-scripts that:\n"
                f"1. Sets default values for the required parameters using set_default_value()\n"
                f"2. Or sets global variables that can be used by the scenario\n\n"
                f"Original error: {error_msg}"
            ) from e
        else:
            raise ValueError(f"Failed to instantiate scenario '{scenario_name}'.\n" f"Error: {error_msg}") from e
    except Exception as e:
        raise ValueError(
            f"Failed to instantiate scenario '{scenario_name}'.\n"
            f"Make sure your initialization scripts properly configure all required parameters.\n"
            f"Error: {e}"
        ) from e

    logger.info(f"Initializing scenario: {scenario.name}")

    # Convert strategy names to enum instances if provided
    strategy_enums = None
    if scenario_strategies:
        strategy_class = scenario_class.get_strategy_class()
        strategy_enums = []
        for strategy_name in scenario_strategies:
            try:
                strategy_enums.append(strategy_class(strategy_name))
            except ValueError:
                available_strategies = [s.value for s in strategy_class]
                raise ValueError(
                    f"Strategy '{strategy_name}' not found in {scenario_class.__name__}.\n"
                    f"Available strategies: {', '.join(available_strategies)}\n"
                    f"Use 'pyrit_scan --list-scenarios' to see available strategies for each scenario."
                ) from None

    # Initialize the scenario with strategy override if provided
    # Note: objective_target and other parameters should be set via initialization scripts
    # using set_default_value() which will be applied by the @apply_defaults decorator
    try:
        if strategy_enums:
            await scenario.initialize_async(scenario_strategies=strategy_enums)  # type: ignore[call-arg]
        else:
            await scenario.initialize_async()  # type: ignore[call-arg]
    except Exception as e:
        raise ValueError(f"Failed to initialize scenario '{scenario_name}'.\n" f"Error: {e}") from e

    logger.info(f"Running scenario: {scenario.name} with {scenario.atomic_attack_count} atomic attacks")

    # Run the scenario
    try:
        result = await scenario.run_async()

        # Print results using ConsoleScenarioResultPrinter
        printer = ConsoleScenarioResultPrinter()
        await printer.print_summary_async(result)

    except Exception as e:
        raise ValueError(f"Failed to run scenario '{scenario_name}'.\n" f"Error: {e}") from e


def _format_wrapped_text(*, text: str, indent: str = "      ", max_width: int = 80) -> None:
    """
    Print text with word wrapping at the specified indentation level.

    Args:
        text (str): The text to wrap and print.
        indent (str): The indentation string to use for each line.
        max_width (int): The maximum line width including indentation.
    """
    words = text.split()
    line = indent

    for word in words:
        if len(line) + len(word) + 1 > max_width:
            print(line)
            line = indent + word
        else:
            if line == indent:
                line += word
            else:
                line += " " + word

    if line.strip():
        print(line)


def _format_scenario_info(*, scenario_info: dict[str, Any]) -> None:
    """
    Print formatted information about a scenario.

    Args:
        scenario_info (dict[str, Any]): Dictionary containing scenario information.
            Expected keys: name, class_name, description, default_strategy, aggregate_strategies, all_strategies
    """
    print(f"\n  {scenario_info['name']}")
    print(f"    Class: {scenario_info['class_name']}")
    print("    Description:")
    _format_wrapped_text(text=scenario_info["description"], indent="      ")

    # Display aggregate strategies if present
    if scenario_info.get("aggregate_strategies"):
        print("    Aggregate Strategies:")
        for strategy in scenario_info["aggregate_strategies"]:
            print(f"      - {strategy}")

    # Display all strategies if present
    if scenario_info.get("all_strategies"):
        print(f"    Available Strategies ({len(scenario_info['all_strategies'])}):")
        strategies_text = ", ".join(scenario_info["all_strategies"])
        _format_wrapped_text(text=strategies_text, indent="      ")

    # Display default strategy if present
    if scenario_info.get("default_strategy"):
        print(f"    Default Strategy: {scenario_info['default_strategy']}")


def _format_initializer_info(*, info: InitializerInfo) -> None:
    """
    Print formatted information about an initializer.

    Args:
        info (InitializerInfo): Dictionary containing initializer information.
    """
    print(f"\n  {info['name']}")
    print(f"    Class: {info['class_name']}")
    print(f"    Name: {info['initializer_name']}")
    print(f"    Execution Order: {info['execution_order']}")

    if info["required_env_vars"]:
        print("    Required Environment Variables:")
        for env_var in info["required_env_vars"]:
            print(f"      - {env_var}")
    else:
        print("    Required Environment Variables: None")

    print("    Description:")
    _format_wrapped_text(text=info["description"], indent="      ")


def _collect_initialization_scripts(*, parsed_args: Namespace) -> list[Path] | None:
    """
    Collect all initialization scripts from built-in initializers and custom scripts.

    Args:
        parsed_args (Namespace): Parsed command-line arguments.

    Returns:
        list[Path] | None: List of script paths, or None if no scripts.

    Raises:
        ValueError: If initializer lookup fails.
        FileNotFoundError: If custom script path does not exist.
    """
    initialization_scripts = []

    # Handle built-in initializers
    if hasattr(parsed_args, "initializers") and parsed_args.initializers:
        registry = InitializerRegistry()
        paths = registry.resolve_initializer_paths(initializer_names=parsed_args.initializers)
        initialization_scripts.extend(paths)

    # Handle custom initialization scripts
    if hasattr(parsed_args, "initialization_scripts") and parsed_args.initialization_scripts:
        paths = InitializerRegistry.resolve_script_paths(script_paths=parsed_args.initialization_scripts)
        initialization_scripts.extend(paths)

    return initialization_scripts if initialization_scripts else None


def _handle_list_scenarios(*, parsed_args: Namespace) -> int:
    """
    Handle the --list-scenarios flag.

    Args:
        parsed_args (Namespace): Parsed command-line arguments.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    initialization_scripts = None

    if hasattr(parsed_args, "initialization_scripts") and parsed_args.initialization_scripts:
        try:
            initialization_scripts = InitializerRegistry.resolve_script_paths(
                script_paths=parsed_args.initialization_scripts
            )

            database = parsed_args.database if hasattr(parsed_args, "database") else SQLITE
            initialize_pyrit(
                memory_db_type=database,
                initialization_scripts=initialization_scripts,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    registry = ScenarioRegistry()
    if initialization_scripts:
        registry.discover_user_scenarios()
    list_scenarios(registry=registry)
    return 0


def _run_scenario(*, parsed_args: Namespace) -> int:
    """
    Run the specified scenario with the given configuration.

    Args:
        parsed_args (Namespace): Parsed command-line arguments.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        # Collect all initialization scripts
        initialization_scripts = _collect_initialization_scripts(parsed_args=parsed_args)

        # Initialize PyRIT
        database = parsed_args.database if hasattr(parsed_args, "database") else SQLITE
        logger.info(f"Initializing PyRIT with database type: {database}")

        initialize_pyrit(
            memory_db_type=database,
            initialization_scripts=initialization_scripts,
        )

        # Create scenario registry and discover user scenarios
        registry = ScenarioRegistry()
        registry.discover_user_scenarios()

        # Get scenario strategies from CLI args if provided
        scenario_strategies = parsed_args.scenario_strategies if hasattr(parsed_args, "scenario_strategies") else None

        # Run the scenario
        asyncio.run(
            run_scenario_async(
                scenario_name=parsed_args.scenario_name,
                registry=registry,
                scenario_strategies=scenario_strategies,
            )
        )
        return 0

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


def main(args=None) -> int:
    """
    Main entry point for the PyRIT scanner CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    # Parse arguments
    try:
        parsed_args = parse_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    # Configure logging
    if parsed_args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Handle discovery flags
    if parsed_args.list_scenarios:
        return _handle_list_scenarios(parsed_args=parsed_args)

    if parsed_args.list_initializers:
        list_initializers()
        return 0

    # Verify scenario was provided
    if not parsed_args.scenario_name:
        print("Error: No scenario specified. Use --help for usage information.")
        return 1

    # Run the scenario
    return _run_scenario(parsed_args=parsed_args)


if __name__ == "__main__":
    sys.exit(main())
