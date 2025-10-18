# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib
import sys
from typing import Any, Literal, Optional, Sequence, Union, get_args

import dotenv

from pyrit.common import path
from pyrit.memory import (
    AzureSQLMemory,
    CentralMemory,
    MemoryInterface,
    SQLiteMemory,
)
from pyrit.common.apply_defaults import reset_default_values

# Import PyRITInitializer for type checking (with TYPE_CHECKING to avoid circular imports)
from typing import TYPE_CHECKING
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


def _load_initializers_from_scripts(*, script_paths: Sequence[Union[str, pathlib.Path]]) -> Sequence["PyRITInitializer"]:
    """
    Load PyRITInitializer instances from external Python files.

    Each script file should contain one or more PyRITInitializer classes and return
    a list of instances via a `get_initializers()` function or by defining a
    variable named `initializers`.

    Args:
        script_paths (Sequence[Union[str, pathlib.Path]]): Sequence of file paths to Python scripts to load.

    Returns:
        Sequence[PyRITInitializer]: List of PyRITInitializer instances loaded from the scripts.

    Raises:
        FileNotFoundError: If a script path does not exist.
        ValueError: If a script path is not a Python file or doesn't contain valid initializers.

    Example:
        Script content (my_custom_init.py):
            from pyrit.setup.initializers.base import PyRITInitializer
            from pyrit.common.apply_defaults import set_default_value
            from pyrit.prompt_target import OpenAIChatTarget

            class MyCustomInitializer(PyRITInitializer):
                @property
                def name(self) -> str:
                    return "My Custom Configuration"

                @property 
                def description(self) -> str:
                    return "Custom OpenAI configuration"

                def initialize(self) -> None:
                    set_default_value(
                        class_type=OpenAIChatTarget,
                        parameter_name="temperature", 
                        value=0.9
                    )

            # Return initializers - either via function:
            def get_initializers():
                return [MyCustomInitializer()]

            # OR via variable:
            # initializers = [MyCustomInitializer()]

        Usage:
            initialize_pyrit(
                memory_db_type="InMemory", 
                initialization_scripts=["my_custom_init.py"]
            )
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
            import importlib.util
            import importlib.machinery
            
            spec = importlib.util.spec_from_file_location(f"init_script_{script.stem}", script)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load initialization script: {script}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Try to get initializers from the module
            script_initializers = []
            
            # Method 1: Look for get_initializers() function
            if hasattr(module, 'get_initializers') and callable(getattr(module, 'get_initializers')):
                script_initializers = module.get_initializers()
                logger.debug(f"Found get_initializers() function in {script.name}")
            
            # Method 2: Look for 'initializers' variable
            elif hasattr(module, 'initializers'):
                script_initializers = module.initializers
                logger.debug(f"Found 'initializers' variable in {script.name}")
            
            else:
                raise ValueError(
                    f"Initialization script {script} must contain either:\n"
                    f"  - A 'get_initializers()' function that returns a list of PyRITInitializer instances, OR\n"
                    f"  - An 'initializers' variable containing a list of PyRITInitializer instances"
                )

            # Validate that all returned items are PyRITInitializer instances
            if not isinstance(script_initializers, (list, tuple)):
                raise ValueError(
                    f"Script {script} must return a list or tuple of PyRITInitializer instances, "
                    f"got {type(script_initializers).__name__}"
                )

            for i, initializer in enumerate(script_initializers):
                if not isinstance(initializer, PyRITInitializer):
                    raise ValueError(
                        f"Script {script} item {i} is not a PyRITInitializer instance. "
                        f"Got {type(initializer).__name__}: {initializer}"
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
    
    # Sort initializers by execution_order (lower numbers first)
    sorted_initializers = sorted(initializers, key=lambda x: x.execution_order)
    
    for initializer in sorted_initializers:
        if not isinstance(initializer, PyRITInitializer):
            raise ValueError(
                f"All initializers must be PyRITInitializer instances. "
                f"Got {type(initializer).__name__}: {initializer}"
            )
        
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

    Example:
        # Using class-based initializers (recommended)
        from pyrit.setup import initialize_pyrit, InitializationPaths
        initialize_pyrit(
            memory_db_type="InMemory",
            initializers=[InitializationPaths.get_simple_initializer()]
        )

        # Using external script files containing PyRITInitializer classes
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=["c:\\myfiles\\custom_initializers.py"]
        )

        # Using both together (all initializers are combined and sorted by execution_order)
        initialize_pyrit(
            memory_db_type="InMemory",
            initializers=[InitializationPaths.get_simple_initializer()],
            initialization_scripts=["c:\\myfiles\\additional_initializers.py"]
        )
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
