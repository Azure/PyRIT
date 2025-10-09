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


class PyRITDefaultValues:

    def __init__(self) -> None:
        self._default_values: dict[DefaultValueScope, object] = {}

    def set_default_value(self, *, scope: DefaultValueScope, value: object) -> None:
        self._default_values[scope] = value

    def get_default_value(
            self,
            *,
            scope: DefaultValueScope, 
            default_value: object
        ) -> object:
        """
        Returns the global default value for the given scope, or the provided default value if none is set.
        """
        if scope in self._default_values:
            return self._default_values[scope]
        return default_value
    
    def get_default_value_for_parameter(
        self,
        *,
        class_type: type,
        parameter_name: str,
        provided_value: Any
    ) -> Any:
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
        
        # Walk up the inheritance hierarchy to find the most specific default
        for cls in inspect.getmro(class_type):
            scope = DefaultValueScope(parameter_name=parameter_name, class_type=cls)
            if scope in self._default_values:
                return self._default_values[scope]
        
        # No default found, return the provided value (None)
        return provided_value


# Global instance for managing default values
_global_default_values = PyRITDefaultValues()


F = TypeVar('F', bound=Callable[..., None])


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
            if param_name == 'self':
                continue
                
            # Get the current value (might be None from defaults or user-provided)
            current_value = bound_args.arguments.get(param_name, None)
            
            # Try to get a default value from the global registry
            default_value = _global_default_values.get_default_value_for_parameter(
                class_type=class_type,
                parameter_name=param_name,
                provided_value=current_value
            )
            
            # Update kwargs with the resolved value
            if param_name in bound_args.arguments:
                kwargs[param_name] = default_value
        
        # Call the original __init__ with updated kwargs
        return init_func(self, *args, **kwargs)
    
    return wrapper  # type: ignore


def set_default_value(*, class_type: type, parameter_name: str, value: Any) -> None:
    """
    Convenience function to set a default value for a specific class and parameter.
    
    Args:
        class_type (type): The class type to set the default for.
        parameter_name (str): The name of the parameter.
        value (Any): The default value to set.
    """
    scope = DefaultValueScope(parameter_name=parameter_name, class_type=class_type)
    _global_default_values.set_default_value(scope=scope, value=value)


def get_global_default_values() -> PyRITDefaultValues:
    """
    Returns the global PyRITDefaultValues instance.
    
    Returns:
        PyRITDefaultValues: The global instance managing default values.
    """
    return _global_default_values

