# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Provides convenient access to initialization configurations.

This module provides both file-based initialization scripts and 
class-based initializers that can be used with initialize_pyrit().

The unified initializers (AIRTInitializer, SimpleInitializer) are
recommended as they provide complete configuration profiles with
clear documentation of requirements.

Example:
    from pyrit.setup import initialize_pyrit
    from pyrit.setup.initializers import AIRTInitializer

    # Recommended: Use unified class-based initializers
    initialize_pyrit(
        memory_db_type="InMemory",
        initializers=[AIRTInitializer()]
    )

    # Legacy: Use initialization scripts
    initialize_pyrit(
        memory_db_type="InMemory", 
        initialization_scripts=InitializationPaths.list_all_airt_paths()
    )
"""

import pathlib
from typing import List

from pyrit.setup.initializers import AIRTInitializer, SimpleInitializer, PyRITInitializer


class InitializationPaths:
    """
    Provides access to initialization script paths.

    This class provides direct access to PyRIT's initialization scripts,
    organized into 'airt' (AI Red Team) and 'simple' categories.

    Attributes:
        airt_converter_initialization: Path to the AIRT converter initialization script.
        airt_scorer_initialization: Path to the AIRT scorer initialization script.
        airt_target_initialization: Path to the AIRT target initialization script.
        simple_converter_initialization: Path to the simple converter initialization script.
        simple_scorer_initialization: Path to the simple scorer initialization script.
        simple_target_initialization: Path to the simple target initialization script.

    Example:
        # Access an initialization script
        path = InitializationPaths.airt_scorer_initialization

        # Use with initialize_pyrit
        initialize_pyrit(
            memory_db_type="InMemory",
            initialization_scripts=[InitializationPaths.airt_scorer_initialization]
        )

        # List all AIRT paths
        airt_paths = InitializationPaths.list_all_airt_paths()

        # List all simple paths
        simple_paths = InitializationPaths.list_all_simple_paths()
    """

    def __init__(self) -> None:
        self._CONFIG_PATH = pathlib.Path(__file__).parent / "initialization_scripts"

    @classmethod
    def list_all_initializers(cls) -> List[PyRITInitializer]:
        """
        Get all available unified initializers.

        Returns:
            List[PyRITInitializer]: List of all available configuration initializers.

        Example:
            # List all available initializers
            initializers = InitializationPaths.list_all_initializers()
            for init in initializers:
                print(f"{init.name}: {init.description}")
        """
        return [SimpleInitializer(), AIRTInitializer()]

    @classmethod
    def get_initializer_info(cls) -> None:
        """
        Print detailed information about all available initializers.

        This method displays configuration requirements and descriptions
        for all available PyRIT initializers.

        Example:
            # Display initializer information
            InitializationPaths.get_initializer_info()
        """
        print("Available PyRIT Initializers:")
        print("=" * 50)
        
        for initializer in cls.list_all_initializers():
            info = initializer.get_info()
            print(f"\n{info['name']}")
            print("-" * len(info['name']))
            print(f"Description: {info['description']}")
            print(f"Profile: {info['profile']}")
            
            if info.get('required_env_vars'):
                print("Required Environment Variables:")
                for env_var in info['required_env_vars']:
                    print(f"  - {env_var}")
            else:
                print("Required Environment Variables: None")
            
            if info.get('notes'):
                print(f"Notes: {info['notes']}")
            
            # Display default values and global variables
            if isinstance(info.get('default_values'), list):
                print(f"Default Values Set ({len(info['default_values'])}):")
                for default in info['default_values']:
                    print(f"  - {default['class']}.{default['parameter']} ({default['value_type']})")
            elif info.get('default_values'):
                print(f"Default Values: {info['default_values']}")
            
            if isinstance(info.get('global_variables'), list):
                print(f"Global Variables Set ({len(info['global_variables'])}):")
                for global_var in info['global_variables']:
                    print(f"  - {global_var['name']} ({global_var['value_type']})")
            elif info.get('global_variables'):
                print(f"Global Variables: {info['global_variables']}")


# Create a singleton instance for easy access
initialization_paths = InitializationPaths()
