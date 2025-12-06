"""API routes for converting text using PyRIT prompt converters"""
import logging
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pyrit.prompt_normalizer import PromptNormalizer, PromptConverterConfiguration
from pyrit.models import SeedPrompt, SeedGroup

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/convert", tags=["convert"])


class ConverterConfig(BaseModel):
    class_name: str
    config: Dict[str, Any]


class ConvertRequest(BaseModel):
    text: str
    converters: List[ConverterConfig]


class ConvertResponse(BaseModel):
    original_text: str
    converted_text: str
    converters_applied: int
    converter_identifiers: List[Dict[str, str]]  # PyRIT converter identifiers


@router.post("/", response_model=ConvertResponse)
async def convert_text(request: ConvertRequest):
    """
    Convert text using a sequence of PyRIT converters.
    Uses PromptNormalizer to ensure consistency with PyRIT's converter tracking.
    Returns both original and converted text.
    """
    if not request.converters:
        return ConvertResponse(
            original_text=request.text,
            converted_text=request.text,
            converters_applied=0,
            converter_identifiers=[]
        )
    
    try:
        # Create a SeedPrompt from the input text
        seed_prompt = SeedPrompt(
            value=request.text,
            data_type="text"
        )
        seed_group = SeedGroup(seeds=[seed_prompt])
        
        # Build converter configurations for PromptNormalizer
        converter_configs = []
        for conv_config in request.converters:
            # Import the converter class
            import pyrit.prompt_converter as converter_module
            if not hasattr(converter_module, conv_config.class_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"Converter {conv_config.class_name} not found"
                )
            
            converter_class = getattr(converter_module, conv_config.class_name)
            
            # Check if converter needs converter_target
            import inspect
            sig = inspect.signature(converter_class.__init__)
            needs_converter_target = 'converter_target' in sig.parameters
            
            # Filter out None values
            filtered_config = {k: v for k, v in conv_config.config.items() if v is not None}
            
            # Inject default target if needed
            if needs_converter_target and 'converter_target' not in filtered_config:
                from pyrit.backend.services.target_registry import TargetRegistry
                try:
                    default_target = TargetRegistry.get_default_attack_target()
                    filtered_config['converter_target'] = default_target
                except Exception as target_error:
                    logger.error(f"Failed to create default target: {target_error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"{conv_config.class_name} requires a converter_target but default target could not be created."
                    )
            
            # Instantiate converter
            try:
                converter = converter_class(**filtered_config)
            except TypeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration for {conv_config.class_name}: {str(e)}"
                )
            
            converter_configs.append(converter)
        
        # Use PromptNormalizer to build the message with converters
        normalizer = PromptNormalizer()
        
        # Create a dummy target (we're not sending, just converting)
        from pyrit.prompt_target import TextTarget
        dummy_target = TextTarget()
        
        # Build the message with converters applied
        converter_configuration = PromptConverterConfiguration(converters=converter_configs)
        message = await normalizer.build_message(
            seed_group=seed_group,
            conversation_id="preview",
            request_converter_configurations=[converter_configuration],
            target=dummy_target,
            labels={},
        )
        
        # Extract converted value and converter identifiers
        if message.message_pieces:
            first_piece = message.message_pieces[0]
            return ConvertResponse(
                original_text=first_piece.original_value,
                converted_text=first_piece.converted_value,
                converters_applied=len(request.converters),
                converter_identifiers=first_piece.converter_identifiers
            )
        else:
            raise HTTPException(status_code=500, detail="No message pieces generated")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in convert endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
