# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import patch

import pytest

from pyrit.datasets.seed_datasets.remote.vlsu_multimodal_dataset import (
    VLSUCategory,
    _VLSUMultimodalDataset,
)
from pyrit.memory import SQLiteMemory
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import SeedDataset


class TestVLSUMultimodalDataset:
    """Unit tests for _VLSUMultimodalDataset."""

    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Set up memory instance for image downloads."""
        memory = SQLiteMemory()
        CentralMemory.set_memory_instance(memory)
        yield
        CentralMemory.set_memory_instance(None)

    def test_dataset_name(self):
        """Test that dataset_name property returns correct value."""
        dataset = _VLSUMultimodalDataset()
        assert dataset.dataset_name == "ml_vlsu"

    def test_init_with_categories(self):
        """Test initialization with category filtering."""
        categories = [VLSUCategory.SLURS_HATE_SPEECH, VLSUCategory.DISCRIMINATION]
        dataset = _VLSUMultimodalDataset(categories=categories)
        assert dataset.categories == categories

    def test_init_with_invalid_categories(self):
        """Test that invalid categories raise ValueError."""
        with pytest.raises(ValueError, match="Invalid VLSU categories"):
            _VLSUMultimodalDataset(categories=["invalid_category"])

    def test_init_with_unsafe_grades(self):
        """Test initialization with custom unsafe grades."""
        dataset = _VLSUMultimodalDataset(unsafe_grades=["unsafe"])
        assert dataset.unsafe_grades == ["unsafe"]

    @pytest.mark.asyncio
    async def test_fetch_dataset_text_only_unsafe(self):
        """Test that text prompt is created when text_grade is unsafe."""
        test_uuid = str(uuid.uuid4())
        mock_data = [
            {
                "prompt": "Unsafe text prompt",
                "web_path": "https://example.com/image.jpg",
                "combined_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "text_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "image_category": "C2: Discrimination and Unequal Treatment",
                "uuid": test_uuid,
                "consensus_text_grade": "unsafe",
                "image_grade": "safe",
                "consensus_combined_grade": "safe",
            }
        ]

        dataset_loader = _VLSUMultimodalDataset()

        with patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.seeds) == 1

            text_prompt = dataset.seeds[0]
            assert text_prompt.data_type == "text"
            assert text_prompt.value == "Unsafe text prompt"
            assert text_prompt.name == "ML-VLSU Text"
            assert text_prompt.harm_categories == [
                "C1: Slurs, Hate Speech, Hate Symbols"
            ]
            assert text_prompt.metadata["prompt_type"] == "text_only"
            assert text_prompt.metadata["text_grade"] == "unsafe"

    @pytest.mark.asyncio
    async def test_fetch_dataset_image_only_unsafe(self):
        """Test that image prompt is created when image_grade is unsafe."""
        test_uuid = str(uuid.uuid4())
        mock_data = [
            {
                "prompt": "Safe text prompt",
                "web_path": "https://example.com/image.jpg",
                "combined_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "text_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "image_category": "C2: Discrimination and Unequal Treatment",
                "uuid": test_uuid,
                "consensus_text_grade": "safe",
                "image_grade": "unsafe",
                "consensus_combined_grade": "safe",
            }
        ]

        dataset_loader = _VLSUMultimodalDataset()

        with (
            patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data),
            patch.object(
                dataset_loader,
                "_fetch_and_save_image_async",
                return_value="/fake/path/image.png",
            ),
        ):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 1

            image_prompt = dataset.seeds[0]
            assert image_prompt.data_type == "image_path"
            assert image_prompt.value == "/fake/path/image.png"
            assert image_prompt.name == "ML-VLSU Image"
            assert image_prompt.harm_categories == [
                "C2: Discrimination and Unequal Treatment"
            ]
            assert image_prompt.metadata["prompt_type"] == "image_only"
            assert image_prompt.metadata["image_grade"] == "unsafe"

    @pytest.mark.asyncio
    async def test_fetch_dataset_combined_unsafe(self):
        """Test that combined prompt is created when combined_grade is unsafe."""
        test_uuid = str(uuid.uuid4())
        mock_data = [
            {
                "prompt": "Safe text that becomes unsafe with image",
                "web_path": "https://example.com/image.jpg",
                "combined_category": "C3: Drug Abuse",
                "text_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "image_category": "C2: Discrimination and Unequal Treatment",
                "uuid": test_uuid,
                "consensus_text_grade": "safe",
                "image_grade": "safe",
                "consensus_combined_grade": "unsafe",
            }
        ]

        dataset_loader = _VLSUMultimodalDataset()

        with (
            patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data),
            patch.object(
                dataset_loader,
                "_fetch_and_save_image_async",
                return_value="/fake/path/image.png",
            ),
        ):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 1

            combined_prompt = dataset.seeds[0]
            assert combined_prompt.data_type == "image_path"
            assert combined_prompt.value == "/fake/path/image.png"
            assert (
                combined_prompt.prompt_text
                == "Safe text that becomes unsafe with image"
            )
            assert combined_prompt.name == "ML-VLSU Combined"
            assert combined_prompt.harm_categories == ["C3: Drug Abuse"]
            assert combined_prompt.metadata["prompt_type"] == "combined"
            assert combined_prompt.metadata["combined_grade"] == "unsafe"
            assert combined_prompt.metadata["text_grade"] == "safe"
            assert combined_prompt.metadata["image_grade"] == "safe"

    @pytest.mark.asyncio
    async def test_fetch_dataset_all_unsafe(self):
        """Test that all three prompts are created when all grades are unsafe."""
        test_uuid = str(uuid.uuid4())
        mock_data = [
            {
                "prompt": "Unsafe text",
                "web_path": "https://example.com/image.jpg",
                "combined_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "text_category": "C2: Discrimination and Unequal Treatment",
                "image_category": "C3: Drug Abuse",
                "uuid": test_uuid,
                "consensus_text_grade": "unsafe",
                "image_grade": "unsafe",
                "consensus_combined_grade": "unsafe",
            }
        ]

        dataset_loader = _VLSUMultimodalDataset()

        with (
            patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data),
            patch.object(
                dataset_loader,
                "_fetch_and_save_image_async",
                return_value="/fake/path/image.png",
            ),
        ):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 3

            prompt_types = [s.metadata["prompt_type"] for s in dataset.seeds]
            assert "text_only" in prompt_types
            assert "image_only" in prompt_types
            assert "combined" in prompt_types

            # Verify each uses correct category
            text_prompt = next(
                s for s in dataset.seeds if s.metadata["prompt_type"] == "text_only"
            )
            assert text_prompt.harm_categories == [
                "C2: Discrimination and Unequal Treatment"
            ]

            image_prompt = next(
                s for s in dataset.seeds if s.metadata["prompt_type"] == "image_only"
            )
            assert image_prompt.harm_categories == ["C3: Drug Abuse"]

            combined_prompt = next(
                s for s in dataset.seeds if s.metadata["prompt_type"] == "combined"
            )
            assert combined_prompt.harm_categories == [
                "C1: Slurs, Hate Speech, Hate Symbols"
            ]

    @pytest.mark.asyncio
    async def test_fetch_dataset_all_safe_no_prompts(self):
        """Test that no prompts are created when all grades are safe."""
        mock_data = [
            {
                "prompt": "Safe text",
                "web_path": "https://example.com/image.jpg",
                "combined_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "text_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "image_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "uuid": str(uuid.uuid4()),
                "consensus_text_grade": "safe",
                "image_grade": "safe",
                "consensus_combined_grade": "safe",
            }
        ]

        dataset_loader = _VLSUMultimodalDataset()

        with patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data):
            with pytest.raises(ValueError, match="SeedDataset cannot be empty."):
                await dataset_loader.fetch_dataset(cache=False)

    @pytest.mark.asyncio
    async def test_category_filtering(self):
        """Test that category filtering works correctly."""
        mock_data = [
            {
                "prompt": "Slur prompt",
                "web_path": "https://example.com/image1.jpg",
                "combined_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "text_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "image_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "uuid": str(uuid.uuid4()),
                "consensus_text_grade": "unsafe",
                "image_grade": "safe",
                "consensus_combined_grade": "safe",
            },
            {
                "prompt": "Discrimination prompt",
                "web_path": "https://example.com/image2.jpg",
                "combined_category": "C2: Discrimination and Unequal Treatment",
                "text_category": "C2: Discrimination and Unequal Treatment",
                "image_category": "C2: Discrimination and Unequal Treatment",
                "uuid": str(uuid.uuid4()),
                "consensus_text_grade": "unsafe",
                "image_grade": "safe",
                "consensus_combined_grade": "safe",
            },
        ]

        dataset_loader = _VLSUMultimodalDataset(
            categories=[VLSUCategory.SLURS_HATE_SPEECH]
        )

        with patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 1
            assert "C1: Slurs" in str(dataset.seeds[0].harm_categories)

    @pytest.mark.asyncio
    async def test_handles_failed_image_downloads(self):
        """Test that dataset handles failed image downloads gracefully."""
        mock_data = [
            {
                "prompt": "Test prompt",
                "web_path": "https://broken-url.com/image.jpg",
                "combined_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "text_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "image_category": "C1: Slurs, Hate Speech, Hate Symbols",
                "uuid": str(uuid.uuid4()),
                "consensus_text_grade": "unsafe",
                "image_grade": "unsafe",
                "consensus_combined_grade": "unsafe",
            }
        ]

        dataset_loader = _VLSUMultimodalDataset()

        with (
            patch.object(dataset_loader, "_fetch_from_url", return_value=mock_data),
            patch.object(
                dataset_loader,
                "_fetch_and_save_image_async",
                side_effect=Exception("Download failed"),
            ),
        ):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            # Should still create text prompt even if image fails
            assert len(dataset.seeds) == 1
            assert dataset.seeds[0].data_type == "text"
            assert dataset.seeds[0].metadata["prompt_type"] == "text_only"
