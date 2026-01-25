# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converters API routes.

Provides endpoints for listing and previewing prompt converters.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.converters import (
    ConversionStep,
    ConverterMetadataResponse,
    PreviewConverterRequest,
    PreviewConverterResponse,
)
from pyrit.backend.services import get_conversation_service, get_registry_service

router = APIRouter(prefix="/converters", tags=["converters"])


@router.get(
    "",
    response_model=List[ConverterMetadataResponse],
)
async def list_converters(
    is_llm_based: Optional[bool] = Query(None, description="Filter by LLM-based converters"),
    is_deterministic: Optional[bool] = Query(None, description="Filter by deterministic converters"),
) -> List[ConverterMetadataResponse]:
    """
    List available converters.

    Returns metadata about all available prompt converters, optionally
    filtered by LLM-based status or determinism.

    Returns:
        List[ConverterMetadataResponse]: List of converter metadata.
    """
    service = get_registry_service()

    return service.get_converters(
        is_llm_based=is_llm_based,
        is_deterministic=is_deterministic,
    )


@router.post(
    "/preview",
    response_model=PreviewConverterResponse,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid converter configuration"},
    },
)
async def preview_converters(request: PreviewConverterRequest) -> PreviewConverterResponse:
    """
    Preview text through a converter pipeline.

    Applies the specified converters in sequence and returns
    intermediate results at each step. Useful for testing converter
    configurations before applying to conversations.

    Returns:
        PreviewConverterResponse: Original content, converted content, and conversion steps.
    """
    service = get_conversation_service()

    try:
        steps_data = await service.preview_converters(request.content, request.converters)

        steps = [
            ConversionStep(
                converter_class=s["converter_type"],
                input=s["input"],
                input_data_type=s.get("input_data_type", "text"),
                output=s["output"],
                output_data_type=s.get("output_type", "text"),
            )
            for s in steps_data
        ]

        final_output = steps[-1].output if steps else request.content
        final_data_type = steps[-1].output_data_type if steps else request.data_type

        return PreviewConverterResponse(
            original_content=request.content,
            converted_content=final_output,
            converted_data_type=final_data_type,
            conversion_chain=steps,
            converter_identifiers=[{"class_name": s.converter_class} for s in steps],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Converter preview failed: {str(e)}",
        )
