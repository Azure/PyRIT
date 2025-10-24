# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib

# Import PyRITInitializer for type checking (with TYPE_CHECKING to avoid circular imports)
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union, get_args

import dotenv

from pyrit.common import path
from pyrit.common.apply_defaults import reset_default_values
from pyrit.memory import (
    AzureSQLMemory,
    CentralMemory,
    MemoryInterface,
    SQLiteMemory,
)

if TYPE_CHECKING:
    from pyrit.setup.initializers.base import PyRITInitializer

logger = logging.getLogger(__name__)

IN_MEMORY = "InMemory"
SQLITE = "SQLite"
AZURE_SQL = "AzureSQL"
MemoryDatabaseType = Literal["InMemory", "SQLite", "AzureSQL"]


def _load_environment_files() -> None:
    """
    Loads the base environment file from .env if it exists,
    and then loads a single .env.local file if it exists, overriding previous values.
    """
    base_file_path = path.HOME_PATH / ".env"
    local_file_path = path.HOME_PATH / ".env.local"

    # Load the base .env file if it exists
    if base_file_path.exists():
        dotenv.load_dotenv(base_file_path, override=True, interpolate=True)
        logger.info(f"Loaded {base_file_path}")
    else:
        dotenv.load_dotenv(verbose=True)

    # Load the .env.local file if it exists, to override base .env values
    if local_file_path.exists():
        dotenv.load_dotenv(local_file_path, override=True, interpolate=True)
        logger.info(f"Loaded {local_file_path}")
    else:
        dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(".env.local"), override=True, verbose=True)


def _load_initializers_from_scripts(
    *, script_paths: Sequence[Union[str, pathlib.Path]]
) -> Sequence["PyRITInitializer"]:
    """
    Load PyRITInitializer instances from external Python files.

    Each script file should contain one or more PyRITInitializer classes. All classes
    that inherit from PyRITInitializer will be automatically discovered and instantiated.

    Args:
        script_paths (Sequence[Union[str, pathlib.Path]]): Sequence of file paths to Python scripts to load.

    Returns:
        Sequence[PyRITInitializer]: List of PyRITInitializer instances loaded from the scripts.

    Raises:
        FileNotFoundError: If a script path does not exist.
        ValueError: If a script path is not a Python file or doesn't contain valid initializers.

    Example:
        Script content should be a subclass of PyRITInitializer e.g. like SimpleInitializer
    """
    # Import here to avoid circular imports
    from pyrit.setup.initializers.base import PyRITInitializer

    loaded_initializers = []

    for script_path in script_paths:
        # Convert to Path object if string
        script = pathlib.Path(script_path)

        # Validate the script exists
        if not script.exists():
            raise FileNotFoundError(f"Initialization script not found: {script}")

        # Validate it's a Python file
        if script.suffix != ".py":
            raise ValueError(f"Initialization script must be a Python file (.py): {script}")

        logger.info(f"Loading initializers from script: {script}")

        # Load the script as a module
        try:
            import importlib.machinery
            import importlib.util

            spec = importlib.util.spec_from_file_location(f"init_script_{script.stem}", script)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load initialization script: {script}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Auto-discover PyRITInitializer subclasses in the module
            script_initializers = []

            # Look for all PyRITInitializer subclasses defined in the module
            for name in dir(module):
                obj = getattr(module, name)
                # Check if it's a class, is a subclass of PyRITInitializer,
                # and is not the base class itself
                if isinstance(obj, type) and issubclass(obj, PyRITInitializer) and obj is not PyRITInitializer:
                    try:
                        # Instantiate the initializer class
                        initializer = obj()
                        script_initializers.append(initializer)
                        logger.debug(f"Found and instantiated {name} in {script.name}")
                    except Exception as e:
                        logger.warning(f"Could not instantiate {name} from {script.name}: {e}")
                        # Continue to try other classes rather than failing completely

            if not script_initializers:
                raise ValueError(
                    f"Initialization script {script} must contain at least one PyRITInitializer subclass. "
                    f"Define a class that inherits from PyRITInitializer."
                )

            loaded_initializers.extend(script_initializers)
            logger.debug(f"Loaded {len(script_initializers)} initializer(s) from {script.name}")

        except Exception as e:
            logger.error(f"Error loading initializers from script {script}: {e}")
            raise

    return loaded_initializers


def _execute_initializers(*, initializers: Sequence["PyRITInitializer"]) -> None:
    """
    Execute PyRITInitializer instances in execution order.

    Initializers are sorted by their execution_order property before execution.
    Lower execution_order values run first.

    Args:
        initializers: Sequence of PyRITInitializer instances to execute.

    Raises:
        ValueError: If an initializer is not a PyRITInitializer instance.
        Exception: If an initializer's validation or initialization fails.
    """
    # Import here to avoid circular imports
    from pyrit.setup.initializers.base import PyRITInitializer

    # Validate all initializers first before sorting
    for initializer in initializers:
        if not isinstance(initializer, PyRITInitializer):
            raise ValueError(
                f"All initializers must be PyRITInitializer instances. "
                f"Got {type(initializer).__name__}: {initializer}"
            )

    # Sort initializers by execution_order (lower numbers first)
    sorted_initializers = sorted(initializers, key=lambda x: x.execution_order)

    for initializer in sorted_initializers:

        logger.info(f"Executing initializer: {initializer.name}")
        logger.debug(f"Description: {initializer.description}")

        try:
            # Validate first
            initializer.validate()

            # Then initialize with tracking to capture what was configured
            initializer.initialize_with_tracking()

            logger.debug(f"Successfully executed initializer: {initializer.name}")

        except Exception as e:
            logger.error(f"Error executing initializer {initializer.name}: {e}")
            raise


def initialize_pyrit(
    memory_db_type: Union[MemoryDatabaseType, str],
    *,
    initialization_scripts: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    initializers: Optional[Sequence["PyRITInitializer"]] = None,
    **memory_instance_kwargs: Any,
) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory_db_type (MemoryDatabaseType): The MemoryDatabaseType string literal which indicates the memory
            instance to use for central memory. Options include "InMemory", "SQLite", and "AzureSQL".
        initialization_scripts (Optional[Sequence[Union[str, pathlib.Path]]]): Optional sequence of Python script paths
            that contain PyRITInitializer classes. Each script must define either a get_initializers() function
            or an 'initializers' variable that returns/contains a list of PyRITInitializer instances.
        initializers (Optional[Sequence[PyRITInitializer]]): Optional sequence of PyRITInitializer instances
            to execute directly. These provide type-safe, validated configuration with clear documentation.
        **memory_instance_kwargs (Optional[Any]): Additional keyword arguments to pass to the memory instance.
    """

    # Handle DuckDB deprecation before validation
    if memory_db_type == "DuckDB":
        logger.warning(
            "DuckDB is no longer supported and has been replaced by SQLite for better compatibility and performance. "
            "Please update your code to use SQLite instead. "
            "For migration guidance, see the SQLite Memory documentation at: "
            "doc/code/memory/1_sqlite_memory.ipynb. "
            "Using in-memory SQLite instead."
        )
        memory_db_type = IN_MEMORY

    _load_environment_files()

    # Reset all default values before executing initialization scripts
    # This ensures a clean state for each initialization
    reset_default_values()

    # Set up memory BEFORE executing initialization scripts
    # This is critical because initialization scripts may instantiate objects
    # (like prompt targets) that require central memory to be initialized
    memory: MemoryInterface

    if memory_db_type == IN_MEMORY:
        logger.info("Using in-memory SQLite database.")
        memory = SQLiteMemory(db_path=":memory:", **memory_instance_kwargs)
    elif memory_db_type == SQLITE:
        logger.info("Using persistent SQLite database.")
        memory = SQLiteMemory(**memory_instance_kwargs)
    elif memory_db_type == AZURE_SQL:
        logger.info("Using AzureSQL database.")
        memory = AzureSQLMemory(**memory_instance_kwargs)
    else:
        raise ValueError(
            f"Memory database type '{memory_db_type}' is not a supported type {get_args(MemoryDatabaseType)}"
        )
    CentralMemory.set_memory_instance(memory)

    # Combine directly provided initializers with those loaded from scripts
    all_initializers = list(initializers) if initializers else []

    # Load additional initializers from scripts
    if initialization_scripts:
        script_initializers = _load_initializers_from_scripts(script_paths=initialization_scripts)
        all_initializers.extend(script_initializers)

    # Execute all initializers (sorted by execution_order)
    if all_initializers:
        _execute_initializers(initializers=all_initializers)
