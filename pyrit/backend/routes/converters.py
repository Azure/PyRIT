# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Converters API routes.

Provides endpoints for:
- Listing converter types (metadata from registry)
- Managing converter instances (runtime objects)
- Previewing converter transformations
"""

from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.converters import (
    ConverterInstance,
    ConverterInstanceListResponse,
    ConverterMetadataResponse,
    ConverterPreviewRequest,
    ConverterPreviewResponse,
    CreateConverterRequest,
    CreateConverterResponse,
)
from pyrit.backend.services import get_registry_service
from pyrit.backend.services.converter_service import get_converter_service

router = APIRouter(prefix="/converters", tags=["converters"])


# ============================================================================
# Converter Types (from registry)
# ============================================================================


@router.get(
    "/types",
    response_model=List[ConverterMetadataResponse],
)
async def list_converter_types(
    is_llm_based: Optional[bool] = Query(None, description="Filter by LLM-based converters"),
    is_deterministic: Optional[bool] = Query(None, description="Filter by deterministic converters"),
) -> List[ConverterMetadataResponse]:
    """
    List available converter types.

    Returns metadata about all available prompt converter types (not instances).
    For instances, use GET /converters/instances.

    Returns:
        List[ConverterMetadataResponse]: List of converter type metadata.
    """
    service = get_registry_service()

    return service.get_converters(
        is_llm_based=is_llm_based,
        is_deterministic=is_deterministic,
    )


# ============================================================================
# Converter Instances (runtime objects)
# ============================================================================


@router.get(
    "/instances",
    response_model=ConverterInstanceListResponse,
)
async def list_converter_instances(
    source: Optional[Literal["initializer", "user"]] = Query(
        None, description="Filter by source (initializer or user)"
    ),
) -> ConverterInstanceListResponse:
    """
    List converter instances.

    Returns all registered converter instances.

    Returns:
        ConverterInstanceListResponse: List of converter instances.
    """
    service = get_converter_service()
    return await service.list_converters(source=source)


@router.post(
    "/instances",
    response_model=CreateConverterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid converter type or parameters"},
        422: {"model": ProblemDetail, "description": "Validation error"},
    },
)
async def create_converter_instance(request: CreateConverterRequest) -> CreateConverterResponse:
    """
    Create a new converter instance.

    Supports nested converters - if params contains a 'converter' key with
    a type/params object, the nested converter will be created first and
    linked to the outer converter.

    Example for SelectiveTextConverter:
    ```json
    {
        "type": "selective_text",
        "params": {
            "pattern": "\\[CONVERT\\]",
            "converter": {
                "type": "base64",
                "params": {}
            }
        }
    }
    ```

    Returns:
        CreateConverterResponse: The created converter instance details.
    """
    service = get_converter_service()

    try:
        return await service.create_converter(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create converter: {str(e)}",
        )


@router.get(
    "/instances/{converter_id}",
    response_model=ConverterInstance,
    responses={
        404: {"model": ProblemDetail, "description": "Converter not found"},
    },
)
async def get_converter_instance(converter_id: str) -> ConverterInstance:
    """
    Get a converter instance by ID.

    Returns:
        ConverterInstance: The converter instance details.
    """
    service = get_converter_service()

    converter = await service.get_converter(converter_id)
    if not converter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Converter '{converter_id}' not found",
        )

    return converter


@router.delete(
    "/instances/{converter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ProblemDetail, "description": "Converter not found"},
    },
)
async def delete_converter_instance(converter_id: str) -> None:
    """
    Delete a converter instance.

    Note: Converters in use by active attacks cannot be deleted.
    """
    service = get_converter_service()

    deleted = await service.delete_converter(converter_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Converter '{converter_id}' not found",
        )


# ============================================================================
# Converter Preview
# ============================================================================


@router.post(
    "/preview",
    response_model=ConverterPreviewResponse,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid converter configuration"},
    },
)
async def preview_conversion(request: ConverterPreviewRequest) -> ConverterPreviewResponse:
    """
    Preview conversion through a converter pipeline.

    Applies converters to the input and returns step-by-step results.
    Can use either converter_ids (existing instances) or inline converters.

    Returns:
        ConverterPreviewResponse: Original, converted values, and conversion steps.
    """
    service = get_converter_service()

    try:
        return await service.preview_conversion(request)
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
