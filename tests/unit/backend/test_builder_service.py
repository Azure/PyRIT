# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.backend.models.builder import BuilderBuildRequest, ReferenceImageRequest
from pyrit.backend.models.converters import (
    ConverterPreviewResponse,
    ConverterTypeListResponse,
    ConverterTypeMetadata,
    PreviewStep,
)
from pyrit.backend.services.builder_service import BuilderService
from pyrit.identifiers import ComponentIdentifier


def _text_converter_type(name: str) -> ConverterTypeMetadata:
    return ConverterTypeMetadata(
        converter_type=name,
        display_name=name,
        description="Prompt rewrite",
        supported_input_types=["text"],
        supported_output_types=["text"],
        parameters=[],
        preview_supported=True,
        preview_unavailable_reason=None,
    )


@pytest.fixture
def builder_service():
    with patch("pyrit.backend.services.builder_service.get_converter_service") as mock_get_converter_service:
        converter_service = MagicMock()
        mock_get_converter_service.return_value = converter_service
        yield BuilderService(), converter_service


@pytest.mark.asyncio
async def test_get_config_returns_prompt_bank_defaults_and_image_capability(builder_service) -> None:
    service, converter_service = builder_service
    converter_service.list_converter_types_async = AsyncMock(
        return_value=ConverterTypeListResponse(
            items=[
                _text_converter_type("VariationConverter"),
                ConverterTypeMetadata(
                    converter_type="AddTextImageConverter",
                    display_name="Add text to image",
                    description="Image workflow",
                    supported_input_types=["image_path"],
                    supported_output_types=["image_path"],
                    parameters=[],
                    preview_supported=True,
                    preview_unavailable_reason=None,
                ),
            ]
        )
    )
    mock_identifier = MagicMock()
    mock_identifier.unique_name = "image-helper"
    mock_target = MagicMock()
    mock_target.get_identifier.return_value = mock_identifier

    with patch.object(service, "_get_reference_image_target", return_value=mock_target):
        response = await service.get_config_async()

    assert response.families
    assert response.presets
    assert response.defaults.default_blocked_words
    assert response.defaults.max_variant_count == 5
    assert response.defaults.multi_variant_converter_types == ["VariationConverter"]
    assert response.capabilities.reference_image_available is True
    assert response.capabilities.reference_image_target_name == "image-helper"


@pytest.mark.asyncio
async def test_build_runs_preset_then_blocked_words_then_converter_then_variants(builder_service) -> None:
    service, converter_service = builder_service
    converter_service.preview_converter_type_async = AsyncMock(
        return_value=ConverterPreviewResponse(
            original_value="rewritten source",
            original_value_data_type="text",
            converted_value="base version",
            converted_value_data_type="text",
            steps=[
                PreviewStep(
                    converter_id="preview",
                    converter_type="VariationConverter",
                    input_value="rewritten source",
                    input_data_type="text",
                    output_value="base version",
                    output_data_type="text",
                )
            ],
        )
    )

    denylist_converter = MagicMock()
    denylist_converter.convert_async = AsyncMock(
        return_value=MagicMock(output_text="rewritten source", output_type="text")
    )
    variation_converter = MagicMock()
    variation_converter.convert_variations_async = AsyncMock(
        return_value=["variation two", "variation three"]
    )

    with (
        patch.object(service, "_get_default_converter_target", return_value=MagicMock()),
        patch("pyrit.backend.services.builder_service.DenylistConverter", return_value=denylist_converter),
        patch("pyrit.backend.services.builder_service.VariationConverter", return_value=variation_converter),
    ):
        response = await service.build_async(
            request=BuilderBuildRequest(
                source_content="",
                source_content_data_type="text",
                converter_type="VariationConverter",
                converter_params={"length_mode": "long"},
                preset_id="single_roleplay_director",
                preset_values={
                    "character_concept": "a masked antihero",
                    "scene_setting": "a rain-soaked alley",
                    "visual_style": "grainy realism",
                    "motion_hook": "a slow crane move",
                    "goal": "menacing escalation",
                },
                avoid_blocked_words=True,
                blocked_words=["kill", "gore"],
                variant_count=3,
            )
        )

    preview_request = converter_service.preview_converter_type_async.await_args.kwargs["request"]

    assert preview_request.original_value == "rewritten source"
    assert response.resolved_source_value == "rewritten source"
    assert [step.stage for step in response.steps] == ["preset", "blocked_words", "converter", "variants"]
    assert [variant.label for variant in response.variants] == ["Base version", "Variation 2", "Variation 3"]
    assert response.warnings == []


@pytest.mark.asyncio
async def test_build_warns_when_blocked_word_helper_is_missing(builder_service) -> None:
    service, converter_service = builder_service
    converter_service.preview_converter_type_async = AsyncMock(
        return_value=ConverterPreviewResponse(
            original_value="manual source",
            original_value_data_type="text",
            converted_value="built output",
            converted_value_data_type="text",
            steps=[
                PreviewStep(
                    converter_id="preview",
                    converter_type="VariationConverter",
                    input_value="manual source",
                    input_data_type="text",
                    output_value="built output",
                    output_data_type="text",
                )
            ],
        )
    )

    with patch.object(service, "_get_default_converter_target", return_value=None):
        response = await service.build_async(
            request=BuilderBuildRequest(
                source_content="manual source",
                source_content_data_type="text",
                converter_type="VariationConverter",
                converter_params={},
                preset_values={},
                avoid_blocked_words=True,
                blocked_words=["kill"],
                variant_count=1,
            )
        )

    assert response.warnings == [
        "Blocked-word avoidance is unavailable because no helper rewrite target is configured."
    ]
    assert [step.stage for step in response.steps] == ["converter"]


@pytest.mark.asyncio
async def test_build_keeps_single_variant_for_non_text_outputs(builder_service) -> None:
    service, converter_service = builder_service
    converter_service.preview_converter_type_async = AsyncMock(
        return_value=ConverterPreviewResponse(
            original_value="/tmp/source.png",
            original_value_data_type="image_path",
            converted_value="/tmp/output.png",
            converted_value_data_type="image_path",
            steps=[
                PreviewStep(
                    converter_id="preview",
                    converter_type="AddTextImageConverter",
                    input_value="/tmp/source.png",
                    input_data_type="image_path",
                    output_value="/tmp/output.png",
                    output_data_type="image_path",
                )
            ],
        )
    )

    response = await service.build_async(
        request=BuilderBuildRequest(
            source_content="/tmp/source.png",
            source_content_data_type="image_path",
            converter_type="AddTextImageConverter",
            converter_params={},
            preset_values={},
            avoid_blocked_words=False,
            blocked_words=[],
            variant_count=3,
        )
    )

    assert len(response.variants) == 1
    assert response.warnings == [
        "Multiple versions are only available for text outputs. The base output was kept."
    ]


@pytest.mark.asyncio
async def test_generate_reference_image_returns_media_url(builder_service) -> None:
    service, _converter_service = builder_service
    mock_target = MagicMock()
    mock_target.get_identifier.return_value = ComponentIdentifier(
        class_name="ImageHelper",
        class_module="tests.unit.backend.test_builder_service",
    )
    mock_message = MagicMock()
    mock_message.get_value.return_value = "/tmp/reference.png"
    mock_target.send_prompt_async = AsyncMock(return_value=[mock_message])

    with patch.object(service, "_get_reference_image_target", return_value=mock_target):
        response = await service.generate_reference_image_async(
            request=ReferenceImageRequest(prompt="a cinematic antihero portrait")
    )

    assert response.image_path == "/tmp/reference.png"
    assert response.image_url == "/api/media?path=%2Ftmp%2Freference.png"
    assert response.target_name.startswith("ImageHelper::")


@pytest.mark.asyncio
async def test_generate_reference_image_raises_when_unavailable(builder_service) -> None:
    service, _converter_service = builder_service

    with patch.object(service, "_get_reference_image_target", return_value=None):
        with pytest.raises(
            ValueError,
            match="Reference image generation is unavailable because no image helper target is configured.",
        ):
            await service.generate_reference_image_async(request=ReferenceImageRequest(prompt="prompt"))
