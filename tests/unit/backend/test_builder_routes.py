# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pyrit.backend.main import app
from pyrit.backend.models.builder import (
    BuilderBuildResponse,
    BuilderCapabilities,
    BuilderConfigResponse,
    BuilderDefaults,
    BuilderPipelineStep,
    BuilderVariant,
    ReferenceImageResponse,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_get_builder_config_returns_prompt_bank_payload(client: TestClient) -> None:
    with patch("pyrit.backend.routes.builder.get_builder_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.get_config_async = AsyncMock(
            return_value=BuilderConfigResponse(
                families=[],
                presets=[],
                defaults=BuilderDefaults(
                    default_blocked_words=["kill", "gore"],
                    max_variant_count=5,
                    multi_variant_converter_types=["VariationConverter"],
                ),
                capabilities=BuilderCapabilities(
                    reference_image_available=False,
                    reference_image_target_name=None,
                ),
            )
        )
        mock_get_service.return_value = mock_service

        response = client.get("/api/builder/config")

    assert response.status_code == 200
    assert response.json()["defaults"]["multi_variant_converter_types"] == ["VariationConverter"]


def test_build_builder_output_returns_bad_request_for_invalid_input(client: TestClient) -> None:
    with patch("pyrit.backend.routes.builder.get_builder_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.build_async = AsyncMock(side_effect=ValueError("Add source content before building output."))
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/builder/build",
            json={
                "source_content": "",
                "source_content_data_type": "text",
                "converter_type": "VariationConverter",
                "converter_params": {},
                "preset_values": {},
                "avoid_blocked_words": False,
                "blocked_words": [],
                "variant_count": 1,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Add source content before building output."


def test_build_builder_output_returns_variants(client: TestClient) -> None:
    with patch("pyrit.backend.routes.builder.get_builder_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.build_async = AsyncMock(
            return_value=BuilderBuildResponse(
                resolved_source_value="source",
                resolved_source_data_type="text",
                converted_value="built",
                converted_value_data_type="text",
                variants=[
                    BuilderVariant(
                        variant_id="base",
                        label="Base version",
                        value="built",
                        data_type="text",
                        kind="base",
                    )
                ],
                steps=[
                    BuilderPipelineStep(
                        stage="converter",
                        title="Converter: VariationConverter",
                        input_value="source",
                        input_data_type="text",
                        output_value="built",
                        output_data_type="text",
                        detail="Applied the selected converter.",
                    )
                ],
                warnings=[],
            )
        )
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/builder/build",
            json={
                "source_content": "source",
                "source_content_data_type": "text",
                "converter_type": "VariationConverter",
                "converter_params": {},
                "preset_values": {},
                "avoid_blocked_words": False,
                "blocked_words": [],
                "variant_count": 1,
            },
        )

    assert response.status_code == 200
    assert response.json()["variants"][0]["label"] == "Base version"


def test_generate_reference_image_returns_bad_request_when_unavailable(client: TestClient) -> None:
    with patch("pyrit.backend.routes.builder.get_builder_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.generate_reference_image_async = AsyncMock(
            side_effect=ValueError("Reference image generation is unavailable because no image helper target is configured.")
        )
        mock_get_service.return_value = mock_service

        response = client.post("/api/builder/reference-image", json={"prompt": "scene prompt"})

    assert response.status_code == 400
    assert "Reference image generation is unavailable" in response.json()["detail"]


def test_generate_reference_image_returns_payload(client: TestClient) -> None:
    with patch("pyrit.backend.routes.builder.get_builder_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.generate_reference_image_async = AsyncMock(
            return_value=ReferenceImageResponse(
                prompt="scene prompt",
                image_path="/tmp/reference.png",
                image_url="/api/media?path=%2Ftmp%2Freference.png",
                target_name="image-helper",
            )
        )
        mock_get_service.return_value = mock_service

        response = client.post("/api/builder/reference-image", json={"prompt": "scene prompt"})

    assert response.status_code == 200
    assert response.json()["image_path"] == "/tmp/reference.png"
