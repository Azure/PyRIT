# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Default value decorator system for PyRIT.

This module provides decorators and utilities for applying default values to class constructors.
It's designed to work with PyRIT's initialization system but is kept in common to avoid circular imports.
"""

import functools
import inspect
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class DefaultValueScope:
    """
    Represents a scope for default values with class type, parameter name, and inheritance rules.

    This class defines the scope where a default value applies, including whether it should
    be inherited by subclasses.
    """

    class_type: Type
    parameter_name: str
    include_subclasses: bool = True

    def __hash__(self) -> int:
        return hash((self.class_type, self.parameter_name, self.include_subclasses))


class GlobalDefaultValues:
    """
    Global registry for default values that can be applied to class constructors.

    This singleton class maintains a registry of default values that can be automatically
    applied to class parameters when using the @apply_defaults decorator.
    """

    def __init__(self) -> None:
        self._default_values: Dict[DefaultValueScope, Any] = {}

    def set_default_value(
        self,
        *,
        class_type: Type,
        parameter_name: str,
        value: Any,
        include_subclasses: bool = True,
    ) -> None:
        """
        Set a default value for a specific class and parameter.

        Args:
            class_type: The class type for which to set the default.
            parameter_name: The name of the parameter to set the default for.
            value: The default value to set.
            include_subclasses: Whether this default should apply to subclasses as well.
        """
        scope = DefaultValueScope(
            class_type=class_type,
            parameter_name=parameter_name,
            include_subclasses=include_subclasses,
        )
        self._default_values[scope] = value
        logger.debug(f"Set default value for {class_type.__name__}.{parameter_name} = {value}")

    def get_default_value(
        self,
        *,
        class_type: Type,
        parameter_name: str,
    ) -> tuple[bool, Any]:
        """
        Get the default value for a specific class and parameter.

        Args:
            class_type: The class type to get the default for.
            parameter_name: The name of the parameter to get the default for.

        Returns:
            Tuple of (found, value) where found indicates if a default was found.
        """
        # First, try exact match
        scope = DefaultValueScope(
            class_type=class_type,
            parameter_name=parameter_name,
            include_subclasses=True,
        )
        if scope in self._default_values:
            return True, self._default_values[scope]

        # Then, check parent classes if include_subclasses is True
        for existing_scope, value in self._default_values.items():
            if (
                existing_scope.parameter_name == parameter_name
                and existing_scope.include_subclasses
                and issubclass(class_type, existing_scope.class_type)
            ):
                return True, value

        return False, None

    def reset_defaults(self) -> None:
        """Reset all default values."""
        self._default_values.clear()
        logger.debug("Reset all default values")

    @property
    def all_defaults(self) -> Dict[DefaultValueScope, Any]:
        """Get a copy of all current default values."""
        return self._default_values.copy()


# Global instance
_global_default_values = GlobalDefaultValues()


def get_global_default_values() -> GlobalDefaultValues:
    """Get the global default values registry."""
    return _global_default_values


def set_default_value(
    *,
    class_type: Type,
    parameter_name: str,
    value: Any,
    include_subclasses: bool = True,
) -> None:
    """
    Set a default value for a specific class and parameter.

    This is a convenience function that delegates to the global default values registry.

    Args:
        class_type: The class type for which to set the default.
        parameter_name: The name of the parameter to set the default for.
        value: The default value to set.
        include_subclasses: Whether this default should apply to subclasses as well.
    """
    _global_default_values.set_default_value(
        class_type=class_type,
        parameter_name=parameter_name,
        value=value,
        include_subclasses=include_subclasses,
    )


def reset_default_values() -> None:
    """Reset all default values in the global registry."""
    _global_default_values.reset_defaults()


def set_global_variable(*, name: str, value: Any) -> None:
    """
    Explicitly sets a global variable in the __main__ module namespace.

    This function provides an alternative to relying on naming conventions for variable exposure.
    Instead of using underscore-prefixed variables that may or may not be exposed based on
    the expose_private_vars parameter, this function explicitly sets variables in the global
    namespace, making the intent clear and the behavior predictable.

    Args:
        name (str): The name of the global variable to set.
        value (Any): The value to assign to the global variable.

    Example:
        # Instead of relying on naming conventions:
        # _helper_config = SomeConfig(...)  # May not be exposed
        # global_config = _helper_config    # Exposed globally

        # Use explicit global variable setting:
        helper_config = SomeConfig(...)
        set_global_variable(name="global_config", value=helper_config)

    Note:
        This function directly modifies the __main__ module's namespace, making the
        variable accessible to code that imports or executes after the initialization
        script runs.
    """

    # Set the variable in the __main__ module's global namespace
    sys.modules["__main__"].__dict__[name] = value


def apply_defaults_to_method(method):
    """
    Decorator that applies default values to a method's parameters.

    This decorator looks up default values for the method's class and applies them
    to parameters that are None or not provided.

    Args:
        method: The method to decorate (typically __init__).

    Returns:
        The decorated method.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Get the class of the instance
        cls = self.__class__

        # Get method signature
        sig = inspect.signature(method)

        # Bind arguments to get parameter names and values
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Apply default values for parameters that are None
        for param_name, param_value in bound_args.arguments.items():
            if param_name == "self":
                continue

            # Only apply defaults if the parameter is None
            if param_value is None:
                found, default_value = _global_default_values.get_default_value(
                    class_type=cls,
                    parameter_name=param_name,
                )
                if found:
                    bound_args.arguments[param_name] = default_value
                    logger.debug(f"Applied default value for {cls.__name__}.{param_name} = {default_value}")

        # Call the original method with updated arguments
        return method(*bound_args.args, **bound_args.kwargs)

    return wrapper


def apply_defaults(method):
    """
    Decorator that applies default values to a class constructor.

    This is an alias for apply_defaults_to_method for backward compatibility.

    Args:
        method: The method to decorate (typically __init__).

    Returns:
        The decorated method.
    """
    return apply_defaults_to_method(method)
