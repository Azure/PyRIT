# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
PyRIT CLI - Command-line interface for running security scenarios.

This module provides the main entry point for the pyrit_scan command.
It supports running scenarios with configurable database backends and
initialization scripts.
"""

import asyncio
import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

from pyrit.cli.scenario_registry import ScenarioRegistry
from pyrit.scenarios.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.setup import IN_MEMORY, SQLITE, AZURE_SQL, initialize_pyrit

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
  pyrit_scan foundry_scenario --initializers simple scenarios.objective_target
  
  # Run with custom initialization scripts
  pyrit_scan encoding_scenario --initialization-scripts ./my_config.py
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
        help="List all available built-in initializers and exit",
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
        help="Built-in initializer names (e.g., simple, airt, scenarios.objective_target)",
    )
    parser.add_argument(
        "--initialization-scripts",
        type=str,
        nargs="+",
        help="Paths to custom Python initialization scripts that configure scenarios and defaults",
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
        print(f"\n  {scenario_info['name']}")
        print(f"    Class: {scenario_info['class_name']}")
        print(f"    Description:")
        # Word wrap the description to 80 characters (accounting for indentation)
        description = scenario_info['description']
        words = description.split()
        line = "      "
        for word in words:
            if len(line) + len(word) + 1 > 74:  # 80 - 6 for indentation
                print(line)
                line = "      " + word
            else:
                if line == "      ":
                    line += word
                else:
                    line += " " + word
        if line.strip():
            print(line)
    
    print("\n" + "=" * 80)
    print(f"\nTotal scenarios: {len(scenarios)}")
    print("\nFor usage information, use: pyrit_scan --help")


def list_initializers() -> None:
    """
    Print all available built-in initializers to the console.
    
    Discovers initializers from pyrit/setup/initializers directory.
    """
    from pyrit.common.path import PYRIT_PATH
    import importlib.util
    import inspect
    from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer
    
    initializers_path = Path(PYRIT_PATH) / "setup" / "initializers"
    
    if not initializers_path.exists():
        print("No initializers directory found.")
        return
    
    # Discover all Python files in the initializers directory
    initializer_info = []
    
    def discover_in_directory(directory: Path, prefix: str = ""):
        """Recursively discover initializers in a directory."""
        for item in directory.iterdir():
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                # Calculate the initializer name (e.g., "simple" or "scenarios.objective_target")
                if prefix:
                    name = f"{prefix}.{item.stem}"
                else:
                    name = item.stem
                
                # Try to load the module and find PyRITInitializer subclasses
                try:
                    spec = importlib.util.spec_from_file_location(f"pyrit.setup.initializers.{name}", item)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find all PyRITInitializer subclasses in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (inspect.isclass(attr) and 
                                issubclass(attr, PyRITInitializer) and 
                                attr != PyRITInitializer):
                                
                                # Instantiate to get name and description
                                try:
                                    instance = attr()
                                    initializer_info.append({
                                        'name': name,
                                        'class_name': attr.__name__,
                                        'initializer_name': instance.name,
                                        'description': instance.description,
                                        'required_env_vars': instance.required_env_vars,
                                        'execution_order': instance.execution_order,
                                    })
                                except Exception:
                                    # Skip initializers that can't be instantiated
                                    pass
                except Exception:
                    # Skip files that can't be loaded
                    pass
            
            elif item.is_dir() and item.name != "__pycache__":
                # Recursively discover in subdirectories
                subdir_prefix = f"{prefix}.{item.name}" if prefix else item.name
                discover_in_directory(item, subdir_prefix)
    
    discover_in_directory(initializers_path)
    
    if not initializer_info:
        print("No initializers found.")
        return
    
    # Sort by execution order, then by name
    initializer_info.sort(key=lambda x: (x['execution_order'], x['name']))
    
    print("\nAvailable Built-in Initializers:")
    print("=" * 80)
    
    for info in initializer_info:
        print(f"\n  {info['name']}")
        print(f"    Class: {info['class_name']}")
        print(f"    Name: {info['initializer_name']}")
        print(f"    Execution Order: {info['execution_order']}")
        
        if info['required_env_vars']:
            print(f"    Required Environment Variables:")
            for env_var in info['required_env_vars']:
                print(f"      - {env_var}")
        else:
            print(f"    Required Environment Variables: None")
        
        print(f"    Description:")
        # Word wrap the description
        description = info['description']
        words = description.split()
        line = "      "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                print(line)
                line = "      " + word
            else:
                if line == "      ":
                    line += word
                else:
                    line += " " + word
        if line.strip():
            print(line)
    
    print("\n" + "=" * 80)
    print(f"\nTotal initializers: {len(initializer_info)}")
    print("\nFor usage information, use: pyrit_scan --help")


async def run_scenario_async(
    *,
    scenario_name: str,
    registry: ScenarioRegistry,
) -> None:
    """
    Run a specific scenario by name.
    
    Args:
        scenario_name (str): Name of the scenario to run.
        registry (ScenarioRegistry): The scenario registry to query.
    
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
    
    # Instantiate the scenario without arguments
    # The scenario should get its configuration from:
    # 1. Default values set by initialization scripts via @apply_defaults
    # 2. Global variables set by initialization scripts
    try:
        scenario = scenario_class()
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
                f"Example initialization script:\n"
                f"  from pyrit.common.apply_defaults import set_default_value\n"
                f"  from pyrit.prompt_target import OpenAIChatTarget\n"
                f"  \n"
                f"  target = OpenAIChatTarget()\n"
                f"  set_default_value('{scenario_class.__name__}', 'objective_target', target)\n\n"
                f"Original error: {error_msg}"
            ) from e
        else:
            raise ValueError(
                f"Failed to instantiate scenario '{scenario_name}'.\n"
                f"Error: {error_msg}"
            ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to instantiate scenario '{scenario_name}'.\n"
            f"Make sure your initialization scripts properly configure all required parameters.\n"
            f"Error: {e}"
        ) from e
    
    logger.info(f"Initializing scenario: {scenario.name}")
    
    # Initialize the scenario (load atomic attacks)
    try:
        await scenario.initialize_async()
    except Exception as e:
        raise ValueError(
            f"Failed to initialize scenario '{scenario_name}'.\n"
            f"Error: {e}"
        ) from e
    
    logger.info(f"Running scenario: {scenario.name} with {scenario.atomic_attack_count} atomic attacks")
    
    # Run the scenario
    try:
        result = await scenario.run_async()
        
        # Print results using ConsoleScenarioResultPrinter
        printer = ConsoleScenarioResultPrinter()
        await printer.print_summary_async(result)
        
    except Exception as e:
        raise ValueError(
            f"Failed to run scenario '{scenario_name}'.\n"
            f"Error: {e}"
        ) from e


def main(args=None) -> int:
    """
    Main entry point for the PyRIT scanner CLI.
    
    Args:
        args: Command-line arguments. If None, uses sys.argv.
    
    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    # Parse arguments first to check for verbose flag
    try:
        parsed_args = parse_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    
    # Configure logging based on verbose flag
    if parsed_args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Suppress most logging unless it's a warning or error
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
    
    # Handle discovery/help flags (these exit after displaying info)
    if parsed_args.list_scenarios:
        # If user provides initialization scripts, we need to initialize PyRIT first
        # to allow discovery of user-defined scenarios
        if hasattr(parsed_args, 'initialization_scripts') and parsed_args.initialization_scripts:
            initialization_scripts = []
            
            for script in parsed_args.initialization_scripts:
                script_path = Path(script)
                if not script_path.is_absolute():
                    script_path = Path.cwd() / script_path
                
                if not script_path.exists():
                    print(f"Error: Initialization script not found: {script_path}")
                    print(f"  Looked in: {script_path.absolute()}")
                    return 1
                
                initialization_scripts.append(script_path)
            
            # Initialize PyRIT to load user scripts
            database = parsed_args.database if hasattr(parsed_args, 'database') else SQLITE
            initialize_pyrit(
                memory_db_type=database,
                initialization_scripts=initialization_scripts,
            )
        
        registry = ScenarioRegistry()
        # Discover user scenarios if initialization scripts were provided
        if hasattr(parsed_args, 'initialization_scripts') and parsed_args.initialization_scripts:
            registry.discover_user_scenarios()
        list_scenarios(registry=registry)
        return 0
    
    if parsed_args.list_initializers:
        list_initializers()
        return 0
    
    # Check if a scenario was provided
    if not parsed_args.scenario_name:
        print("Error: No scenario specified. Use --help for usage information.")
        return 1
    
    try:
        # Run the scenario
        initialization_scripts = []
        
        # Handle built-in initializers
        if hasattr(parsed_args, 'initializers') and parsed_args.initializers:
            from pyrit.common.path import PYRIT_PATH
            initializers_path = Path(PYRIT_PATH) / "setup" / "initializers"
            
            for initializer_name in parsed_args.initializers:
                # Convert dot notation to path (e.g., "scenarios.objective_target" -> "scenarios/objective_target.py")
                # Also support simple names (e.g., "simple" -> "simple.py")
                name_parts = initializer_name.split('.')
                initializer_file = initializers_path
                for part in name_parts:
                    initializer_file = initializer_file / part
                initializer_file = initializer_file.with_suffix('.py')
                
                if not initializer_file.exists():
                    print(f"Error: Built-in initializer '{initializer_name}' not found.")
                    print(f"Available initializers: simple, airt, scenarios.objective_target, scenarios.objective_list")
                    return 1
                
                initialization_scripts.append(initializer_file)
        
        # Handle custom initialization scripts
        if hasattr(parsed_args, 'initialization_scripts') and parsed_args.initialization_scripts:
            for script in parsed_args.initialization_scripts:
                script_path = Path(script)
                
                # If path is not absolute, resolve it relative to current working directory
                if not script_path.is_absolute():
                    script_path = Path.cwd() / script_path
                
                # Validate that script exists
                if not script_path.exists():
                    print(f"Error: Initialization script not found: {script_path}")
                    print(f"  Looked in: {script_path.absolute()}")
                    return 1
                
                initialization_scripts.append(script_path)
        
        # Convert to None if empty
        initialization_scripts = initialization_scripts if initialization_scripts else None
        
        database = parsed_args.database if hasattr(parsed_args, 'database') else SQLITE
        logger.info(f"Initializing PyRIT with database type: {database}")
        
        initialize_pyrit(
            memory_db_type=database,
            initialization_scripts=initialization_scripts,
        )
        
        # Create scenario registry
        registry = ScenarioRegistry()
        
        # Discover user scenarios from initialization scripts
        registry.discover_user_scenarios()
        
        asyncio.run(run_scenario_async(
            scenario_name=parsed_args.scenario_name,
            registry=registry,
        ))
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

