# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario signature enforcement

This module provides mechanisms to ensure all Scenario subclasses maintain
consistent parameter signatures, particularly for objective_target and
scenario_strategies which are required for CLI integration.
"""

import inspect
import logging
from abc import ABCMeta
from typing import Any, Dict, Type


logger = logging.getLogger(__name__)


class ScenarioSignatureEnforcer(ABCMeta):
    """
    Metaclass that enforces standardized parameter signatures on scenario subclasses.

    This metaclass validates that all concrete Scenario subclasses have the required
    parameters in their __init__ method signatures. This is critical for CLI integration
    where the CLI needs to inject standard parameters like objective_target and
    scenario_strategies into any scenario class.

    The enforcement happens at class definition time (when the class is created),
    catching signature issues immediately rather than at runtime.

    Required parameters that must be present in every scenario __init__:
        - objective_target: The target system to attack
        - scenario_strategies: List of strategies to execute
        - max_concurrency: Maximum concurrent operations
        - max_retries: Number of automatic retries on failure
        - memory_labels: Labels for memory tracking
    """

    # Required parameters that MUST exist with these exact names in every scenario
    REQUIRED_PARAMETERS: Dict[str, str] = {
        "objective_target": "PromptTarget",
        "scenario_strategies": "Optional[Sequence[ScenarioStrategy | ScenarioCompositeStrategy]]",
        "max_concurrency": "int",
        "max_retries": "int",
        "memory_labels": "Optional[Dict[str, str]]",
    }

    def __new__(mcs, name: str, bases: tuple, namespace: Dict[str, Any]) -> Type:
        """
        Create a new class and validate its __init__ signature.

        Args:
            name (str): The name of the class being created.
            bases (tuple): Base classes of the class being created.
            namespace (Dict[str, Any]): The class namespace dictionary.

        Returns:
            Type: The created class.

        Raises:
            TypeError: If a concrete scenario subclass has an invalid signature.
        """
        cls = super().__new__(mcs, name, bases, namespace)

        # Only validate concrete scenario subclasses (skip abstract base classes)
        if mcs._should_validate_class(cls, bases):
            mcs._validate_init_signature(cls)

        return cls

    @classmethod
    def _should_validate_class(mcs, cls: Type, bases: tuple) -> bool:
        """
        Determine if a class should have its signature validated.

        Args:
            cls (Type): The class to check.
            bases (tuple): Base classes of the class.

        Returns:
            bool: True if the class should be validated, False otherwise.
        """
        # Skip if this is the base Scenario class itself
        if cls.__name__ == "Scenario":
            return False

        # Skip abstract classes (marked with __abstract__ attribute)
        if getattr(cls, "__abstract__", False):
            return False

        # Skip test helper classes (test fixtures defined in other test files)
        # BUT: Do NOT skip classes in test_scenario_signature_enforcer.py since those
        # are specifically testing the validation logic itself
        module = cls.__module__
        if module and ("test_" in module or ".tests." in module):
            # Allow validation for test_scenario_signature_enforcer specifically
            if "test_scenario_signature_enforcer" not in module:
                return False

        # Only validate if it's a subclass of a class named "Scenario"
        # (avoiding circular import by checking class names)
        has_scenario_base = any(base.__name__ == "Scenario" for base in bases)
        if not has_scenario_base:
            return False

        # Only validate if the class defines its own __init__
        if "__init__" not in cls.__dict__:
            return False

        return True

    @classmethod
    def _validate_init_signature(mcs, cls: Type) -> None:
        """
        Validate that the class __init__ has all required parameters with correct names.

        This catches the problem where a developer might rename objective_target to
        something else (like my_target), which would break CLI integration.

        Args:
            cls (Type): The class to validate.

        Raises:
            TypeError: If required parameters are missing or have incorrect configuration.
        """
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot validate {cls.__name__}.__init__() signature for CLI compatibility. "
                f"Unable to inspect method signature: {e}"
            ) from e

        # Get all parameters except 'self'
        params = {name: param for name, param in sig.parameters.items() if name != "self"}

        # Check that all required parameters exist with correct names
        missing_params = []
        for required_name, expected_type in mcs.REQUIRED_PARAMETERS.items():
            if required_name not in params:
                missing_params.append(f"  - {required_name}: {expected_type}")

        if missing_params:
            params_list = ", ".join(params.keys())
            raise TypeError(
                f"\n{cls.__name__}.__init__() missing required parameter(s) for CLI compatibility:\n"
                + "\n".join(missing_params)
                + f"\n\nFound parameters: {params_list}\n"
                + f"\nAll scenarios must include these exact parameter names to support CLI operations.\n"
                + f"Example: pyrit run {cls.__name__.lower()} --objective_target <target> --scenario_strategies "
                + "<strategies>"
            )

        logger.debug(f"{cls.__name__} signature validation passed")


def validate_scenario_signature(scenario_class: Type) -> None:
    """
    Manually validate a scenario class signature.

    This function is primarily used for testing purposes, as the metaclass automatically
    validates all scenario classes at definition time. It can be useful for:
    - Testing validation logic on classes marked as abstract
    - Validating dynamically created classes
    - Debugging signature issues

    Args:
        scenario_class (Type): The scenario class to validate.

    Raises:
        TypeError: If the scenario class has an invalid signature.

    Example:
        >>> from pyrit.scenarios import EncodingScenario
        >>> validate_scenario_signature(EncodingScenario)
        # Passes silently if valid, raises TypeError if invalid

    Note:
        In normal usage, you don't need to call this function - the metaclass
        automatically validates all concrete scenario subclasses at class definition time.
    """
    ScenarioSignatureEnforcer._validate_init_signature(scenario_class)
