"""
API routes for prompt converters
"""
import inspect
import logging
from typing import Any, Dict, List, Optional, Union, get_type_hints, get_origin, get_args
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import pyrit.prompt_converter as converter_module

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/converters", tags=["converters"])


class ConverterParameter(BaseModel):
    name: str
    type: str
    required: bool
    description: Optional[str] = None
    default: Optional[Any] = None
    enum_values: Optional[List[str]] = None


class ConverterInfo(BaseModel):
    name: str
    class_name: str
    description: Optional[str] = None
    parameters: List[ConverterParameter]
    uses_llm: bool = False


def get_python_type_info(param_type: Any) -> tuple[str, Optional[List[str]]]:
    """
    Convert Python type hints to UI type strings
    Returns: (type_string, options_list)
    """
    # Handle None type
    if param_type is type(None):
        return ("str", None)
    
    # Get the origin for generic types (like Optional, Union, List)
    origin = get_origin(param_type)
    
    # Handle Optional[T] which is Union[T, None]
    if origin is Union:
        args = get_args(param_type)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            param_type = non_none_args[0]
            origin = get_origin(param_type)
    
    # Handle basic types
    if param_type is bool or param_type == bool:
        return ("bool", None)
    elif param_type is int or param_type == int:
        return ("int", None)
    elif param_type is float or param_type == float:
        return ("float", None)
    elif param_type is str or param_type == str:
        return ("str", None)
    
    # Handle List[str] as enum (if it's a Literal, it would be handled separately)
    if origin is list:
        return ("str", None)  # Default to string for lists
    
    # Check if it's a PromptChatTarget or similar
    type_str = str(param_type)
    if "Target" in type_str or "ChatTarget" in type_str:
        return ("chat_target", None)
    
    # Default to string for unknown types
    return ("str", None)


def extract_converter_info(converter_class) -> Optional[ConverterInfo]:
    """
    Extract metadata from a converter class
    """
    try:
        # Skip abstract base classes and non-converter classes
        if not hasattr(converter_class, '__init__'):
            return None
        
        class_name = converter_class.__name__
        
        # Skip base classes
        if class_name in ['PromptConverter', 'ConverterResult', 'LLMGenericTextConverter', 'WordLevelConverter']:
            return None
        
        # Get docstring
        description = inspect.getdoc(converter_class)
        if description:
            # Take only the first line
            description = description.split('\n')[0].strip()
        
        # Get __init__ signature
        try:
            sig = inspect.signature(converter_class.__init__)
        except (ValueError, TypeError):
            return None
        
        # Get type hints
        try:
            type_hints = get_type_hints(converter_class.__init__)
        except Exception:
            type_hints = {}
        
        parameters = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' and any internal parameters
            if param_name in ['self', 'args', 'kwargs']:
                continue
            
            # Get type from hints or annotation
            param_type = type_hints.get(param_name, param.annotation)
            
            # Convert to UI type
            type_str, options = get_python_type_info(param_type)
            
            # Skip chat_target parameters - backend will handle these
            if type_str == "chat_target":
                continue
            
            # Check if parameter has a default value
            has_default = param.default != inspect.Parameter.empty
            default_value = param.default if has_default else None
            
            # Convert default to serializable format
            if default_value is not None and not isinstance(default_value, (str, int, float, bool)):
                default_value = None
            
            parameters.append(
                ConverterParameter(
                    name=param_name,
                    type=type_str,
                    required=not has_default,
                    default=default_value,
                    enum_values=options,
                    description=None
                )
            )
        
        # Create a more readable name from class name
        readable_name = class_name.replace('Converter', '').replace('_', ' ')
        
        # Detect if converter uses LLM
        uses_llm = False
        try:
            # Check if it inherits from LLMGenericTextConverter
            base_classes = inspect.getmro(converter_class)
            for base in base_classes:
                if 'LLMGenericTextConverter' in base.__name__ or 'LLM' in class_name:
                    uses_llm = True
                    break
            
            # Also check if any parameter is a chat_target (filtered out but indicates LLM usage)
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'args', 'kwargs']:
                    continue
                param_type = type_hints.get(param_name, param.annotation)
                type_str, _ = get_python_type_info(param_type)
                if type_str == "chat_target":
                    uses_llm = True
                    break
        except Exception:
            pass
        
        return ConverterInfo(
            name=readable_name,
            class_name=class_name,
            description=description,
            parameters=parameters,
            uses_llm=uses_llm
        )
    
    except Exception as e:
        logger.warning(f"Failed to extract info for {converter_class}: {e}")
        return None


@router.get("/", response_model=List[ConverterInfo])
async def get_converters():
    """
    Get list of available prompt converters with their parameter metadata
    """
    converters = []
    
    # Iterate through all classes in the converter module
    for name in dir(converter_module):
        obj = getattr(converter_module, name)
        
        # Check if it's a class and looks like a converter
        if inspect.isclass(obj) and name.endswith('Converter'):
            converter_info = extract_converter_info(obj)
            if converter_info:
                converters.append(converter_info)
    
    # Sort by name for consistent ordering
    converters.sort(key=lambda x: x.name)
    
    logger.info(f"Found {len(converters)} converters")
    return converters


@router.get("/{class_name}", response_model=ConverterInfo)
async def get_converter_info(class_name: str):
    """
    Get detailed information about a specific converter
    """
    # Find the converter class
    if not hasattr(converter_module, class_name):
        raise HTTPException(status_code=404, detail=f"Converter {class_name} not found")
    
    converter_class = getattr(converter_module, class_name)
    converter_info = extract_converter_info(converter_class)
    
    if not converter_info:
        raise HTTPException(status_code=404, detail=f"Could not extract info for {class_name}")
    
    return converter_info
