# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


@dataclass(frozen=True)
class DefaultValueScope:
    """
    Determines the scope for the default value
    """

    # The name of the parameter to apply
    parameter_name: str

    # The most specific class name for the default value to apply
    class_type: type

    # Whether this default should apply to subclasses
    include_subclasses: bool = True


class PyRITDefaultValues:

    def __init__(self) -> None:
        self._default_values: dict[DefaultValueScope, object] = {}

    def set_default_value(self, *, scope: DefaultValueScope, value: object) -> None:
        self._default_values[scope] = value

    def reset_default_values(self) -> None:
        """
        Clears all configured default values.

        This method removes all default values that have been set, returning the
        PyRITDefaultValues instance to its initial empty state. This is useful for
        test isolation or when re-initializing PyRIT with a fresh configuration.
        """
        self._default_values.clear()

    def get_default_value(self, *, scope: DefaultValueScope, default_value: object) -> object:
        """
        Returns the global default value for the given scope, or the provided default value if none is set.
        """
        if scope in self._default_values:
            return self._default_values[scope]
        return default_value

    def get_default_value_for_parameter(self, *, class_type: type, parameter_name: str, provided_value: Any) -> Any:
        """
        Returns the default value for a specific class and parameter name.

        This method checks the inheritance hierarchy to find the most specific default value.

        Args:
            class_type (type): The class type to look up defaults for.
            parameter_name (str): The name of the parameter.
            provided_value (Any): The value that was provided by the user.

        Returns:
            Any: The default value if one exists and no value was provided, otherwise the provided value.
        """
        # If a value was provided, use it
        if provided_value is not None:
            return provided_value

        # First, check for an exact match with include_subclasses=False
        exact_scope = DefaultValueScope(parameter_name=parameter_name, class_type=class_type, include_subclasses=False)
        if exact_scope in self._default_values:
            return self._default_values[exact_scope]

        # Walk up the inheritance hierarchy to find defaults with include_subclasses=True
        for cls in inspect.getmro(class_type):
            scope = DefaultValueScope(parameter_name=parameter_name, class_type=cls, include_subclasses=True)
            if scope in self._default_values:
                return self._default_values[scope]

        # No default found, return the provided value (None)
        return provided_value


# Global instance for managing default values
_global_default_values = PyRITDefaultValues()


F = TypeVar("F", bound=Callable[..., None])


def apply_defaults(init_func: F) -> F:
    """
    Decorator that automatically applies default values to __init__ parameters.

    This decorator inspects the __init__ method's parameters and applies any configured
    default values from the global PyRITDefaultValues instance. It walks up the class
    inheritance hierarchy to find the most specific default value for each parameter.

    Usage:
        @apply_defaults
        def __init__(self, *, temperature: Optional[float] = None, top_p: Optional[float] = None):
            # By the time we get here, defaults have been applied if they were None
            self._temperature = temperature
            self._top_p = top_p

    Note:
        - Only applies defaults when the provided value is None
        - Respects explicitly provided values (including False, 0, "", etc.)
        - Checks inheritance hierarchy for most specific defaults
    """

    @functools.wraps(init_func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        # Get the signature of the __init__ method
        sig = inspect.signature(init_func)

        # Get the class type from self
        class_type = type(self)

        # Bind the provided arguments to get actual parameter values
        bound_args = sig.bind_partial(self, *args, **kwargs)
        bound_args.apply_defaults()

        # For each parameter (except self), check if we should apply a default
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get the current value (might be None from defaults or user-provided)
            current_value = bound_args.arguments.get(param_name, None)

            # Try to get a default value from the global registry
            default_value = _global_default_values.get_default_value_for_parameter(
                class_type=class_type, parameter_name=param_name, provided_value=current_value
            )

            # Update kwargs with the resolved value if a default was found
            # This applies even if the parameter wasn't originally provided
            if default_value is not None:
                kwargs[param_name] = default_value

        # Call the original __init__ with updated kwargs
        return init_func(self, *args, **kwargs)

    return wrapper  # type: ignore


M = TypeVar("M", bound=Callable[..., Any])


def apply_defaults_to_method(method_func: M) -> M:
    """
    Decorator that automatically applies default values to method parameters.

    This decorator is similar to apply_defaults but works with static methods, class methods,
    or any callable where the first parameter is not 'self'. It inspects the method's
    parameters and applies any configured default values from the global PyRITDefaultValues instance.

    Usage:
        class MyFactory:
            @staticmethod
            @apply_defaults_to_method
            def create_instance(*, param1: Optional[str] = None, param2: Optional[int] = None):
                # By the time we get here, defaults have been applied if they were None
                return MyClass(param1=param1, param2=param2)

    Note:
        - Only applies defaults when the provided value is None
        - Respects explicitly provided values (including False, 0, "", etc.)
        - For static methods, uses the class type from the method's __qualname__
        - For regular functions, uses the function itself as the class type
    """

    @functools.wraps(method_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get the signature of the method
        sig = inspect.signature(method_func)

        # Determine the class type for default value lookup
        # For static methods, parse the class from __qualname__
        # For regular functions, use the function itself
        if hasattr(method_func, "__qualname__") and "." in method_func.__qualname__:
            # This is a method (static or class method)
            # Parse class name from qualname like "ClassName.method_name"
            class_name = method_func.__qualname__.rsplit(".", 1)[0]
            # Try to get the class from the function's globals
            class_type = method_func.__globals__.get(class_name)
            if class_type is None:
                # If we can't find it, we can't apply defaults for this context
                return method_func(*args, **kwargs)
        else:
            # Regular function - use the function itself as the "class type"
            class_type = method_func

        # Bind the provided arguments to get actual parameter values
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # For each parameter, check if we should apply a default
        for param_name, param in sig.parameters.items():
            # Get the current value (might be None from defaults or user-provided)
            current_value = bound_args.arguments.get(param_name, None)

            # Try to get a default value from the global registry
            default_value = _global_default_values.get_default_value_for_parameter(
                class_type=class_type, parameter_name=param_name, provided_value=current_value
            )

            # Update kwargs with the resolved value
            if param_name in bound_args.arguments:
                kwargs[param_name] = default_value

        # Call the original method with updated kwargs
        return method_func(*args, **kwargs)

    return wrapper  # type: ignore


def set_default_value(*, class_type: type, parameter_name: str, value: Any, include_subclasses: bool = True) -> None:
    """
    Convenience function to set a default value for a specific class and parameter.

    Args:
        class_type (type): The class type to set the default for.
        parameter_name (str): The name of the parameter.
        value (Any): The default value to set.
        include_subclasses (bool): If True, the default applies to subclasses.
            If False, only applies to the exact class. Defaults to True.
    """
    scope = DefaultValueScope(
        parameter_name=parameter_name, class_type=class_type, include_subclasses=include_subclasses
    )
    _global_default_values.set_default_value(scope=scope, value=value)


def get_global_default_values() -> PyRITDefaultValues:
    """
    Returns the global PyRITDefaultValues instance.

    Returns:
        PyRITDefaultValues: The global instance managing default values.
    """
    return _global_default_values


def reset_default_values() -> None:
    """
    Clears all configured default values from the global registry.

    This function removes all default values that have been set using set_default_value(),
    returning the system to its initial state with no configured defaults. This is useful
    for test isolation or when re-initializing PyRIT with a fresh configuration.

    Example:
        # Set some defaults
        set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)
        set_default_value(class_type=OpenAIChatTarget, parameter_name="top_p", value=0.9)

        # Clear all defaults
        reset_default_values()
    """
    _global_default_values.reset_default_values()
