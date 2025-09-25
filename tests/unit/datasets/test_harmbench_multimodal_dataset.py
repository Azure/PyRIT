# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets.harmbench_multimodal_dataset import fetch_harmbench_multimodal_dataset_async
from pyrit.models import SeedPromptDataset


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
class TestHarmBenchMultimodalDataset:
    @patch("pyrit.datasets.harmbench_multimodal_dataset._fetch_and_save_image_async")
    @patch("pyrit.datasets.harmbench_multimodal_dataset.fetch_examples")
    async def test_fetching_with_valid_inputs(self, mock_fetch, mock_fetch_image):
        mock_examples = [
            {
                "Behavior": "Test harmful behavior with image",
                "BehaviorID": "test_behavior_1",
                "SemanticCategory": "harmful_category",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "test_image.png",
                "ImageDescription": "A test image description",
                "RedactedImageDescription": "A redacted test image description",
            },
            {
                "Behavior": "Another harmful behavior",
                "BehaviorID": "test_behavior_2",
                "SemanticCategory": "illegal_activity",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "another_image.jpg",
                "ImageDescription": "Another image description",
                "RedactedImageDescription": "Another redacted description",
            },
        ]

        mock_fetch.return_value = mock_examples
        mock_fetch_image.side_effect = [
            "/dbdata/seed-prompt-entries/images/harmbench_test_behavior_1.png",
            "/dbdata/seed-prompt-entries/images/harmbench_test_behavior_2.png",
        ]

        result = await fetch_harmbench_multimodal_dataset_async()

        assert isinstance(result, SeedPromptDataset)
        assert len(result.prompts) == 4  # 2 text + 2 image prompts

        text_prompts = [p for p in result.prompts if p.data_type == "text"]
        image_prompts = [p for p in result.prompts if p.data_type == "image_path"]
        assert len(text_prompts) == 2
        assert len(image_prompts) == 2

        # Verify each text prompt has corresponding image prompt with same group_id
        for text_prompt in text_prompts:
            matching_image = next(p for p in image_prompts if p.prompt_group_id == text_prompt.prompt_group_id)
            assert matching_image is not None
            assert text_prompt.prompt_group_id == matching_image.prompt_group_id

        # Test that the prompts have correct properties set

        # Text properties
        assert text_prompts[0].value == "Test harmful behavior with image"
        assert text_prompts[0].data_type == "text"
        assert text_prompts[0].name == "HarmBench Multimodal Text - test_behavior_1"
        assert text_prompts[0].dataset_name == "HarmBench Multimodal Examples"
        assert text_prompts[0].harm_categories == ["harmful_category"]
        assert (
            text_prompts[0].description
            == "A text prompt from the HarmBench multimodal dataset, BehaviorID: test_behavior_1"
        )
        assert text_prompts[0].sequence == 0
        assert text_prompts[0].metadata == {"behavior_id": "test_behavior_1"}

        # Image properties
        assert image_prompts[1].value == "/dbdata/seed-prompt-entries/images/harmbench_test_behavior_2.png"
        assert image_prompts[1].data_type == "image_path"
        assert image_prompts[1].name == "HarmBench Multimodal Image - test_behavior_2"
        assert image_prompts[1].dataset_name == "HarmBench Multimodal Examples"
        assert image_prompts[1].harm_categories == ["illegal_activity"]
        assert (
            image_prompts[1].description
            == "An image prompt from the HarmBench multimodal dataset, BehaviorID: test_behavior_2"
        )
        assert image_prompts[1].sequence == 0
        expected_metadata = {
            "behavior_id": "test_behavior_2",
            "image_description": "Another image description",
            "redacted_image_description": "Another redacted description",
            "original_image_url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/multimodal_behavior_images/another_image.png",
        }
        assert image_prompts[1].metadata == expected_metadata

    @patch("pyrit.datasets.harmbench_multimodal_dataset._fetch_and_save_image_async")
    @patch("pyrit.datasets.harmbench_multimodal_dataset.fetch_examples")
    async def test_fetching_with_missing_required_keys(self, mock_fetch, mock_fetch_image):
        mock_examples = [
            {
                "Behavior": "Test behavior",
                "BehaviorID": "test_id",
                "FunctionalCategory": "multimodal",
                # Missing SemanticCategory and ImageFileName
            }
        ]

        mock_fetch.return_value = mock_examples
        with pytest.raises(ValueError, match="Missing keys in example"):
            await fetch_harmbench_multimodal_dataset_async()

    @patch("pyrit.datasets.harmbench_multimodal_dataset._fetch_and_save_image_async")
    @patch("pyrit.datasets.harmbench_multimodal_dataset.fetch_examples")
    async def test_fetching_with_missing_optional_fields(self, mock_fetch, mock_fetch_image):
        mock_examples = [
            {
                "Behavior": "Test behavior",
                "BehaviorID": "test_optional",
                "SemanticCategory": "test_category",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "test_optional.png",
                # Missing optional fields: ImageDescription, RedactedImageDescription
            }
        ]

        mock_fetch.return_value = mock_examples
        mock_fetch_image.return_value = "/dbdata/seed-prompt-entries/images/harmbench_test_optional.png"

        result = await fetch_harmbench_multimodal_dataset_async()

        assert isinstance(result, SeedPromptDataset)
        assert len(result.prompts) == 2

        # Verify image prompt handles missing optional fields
        image_prompt = next(p for p in result.prompts if p.data_type == "image_path")
        assert image_prompt.metadata["image_description"] == ""
        assert image_prompt.metadata["redacted_image_description"] == ""

    @patch("pyrit.datasets.harmbench_multimodal_dataset._fetch_and_save_image_async")
    @patch("pyrit.datasets.harmbench_multimodal_dataset.fetch_examples")
    async def test_fetching_with_empty_examples(self, mock_fetch, mock_fetch_image):
        mock_fetch.return_value = []

        with pytest.raises(ValueError, match="SeedPromptDataset cannot be empty"):
            await fetch_harmbench_multimodal_dataset_async()

    @patch("pyrit.datasets.harmbench_multimodal_dataset._fetch_and_save_image_async")
    @patch("pyrit.datasets.harmbench_multimodal_dataset.fetch_examples")
    async def test_filtering_out_non_multimodal_examples(self, mock_fetch, mock_fetch_image):
        mock_examples = [
            {
                "Behavior": "Text only behavior",
                "BehaviorID": "text_only",
                "SemanticCategory": "harmful",
                "FunctionalCategory": "text_generation",  # Non-multimodal
                "ImageFileName": "unused.png",
            },
            {
                "Behavior": "Multimodal behavior",
                "BehaviorID": "multimodal_id",
                "SemanticCategory": "harmful",
                "FunctionalCategory": "multimodal",  # This should be included
                "ImageFileName": "valid.png",
            },
        ]

        mock_fetch.return_value = mock_examples
        mock_fetch_image.return_value = "/dbdata/seed-prompt-entries/images/harmbench_multimodal_id.png"

        result = await fetch_harmbench_multimodal_dataset_async()

        assert len(result.prompts) == 2  # one example (1 text + 1 image)
        assert all(p.metadata["behavior_id"] == "multimodal_id" for p in result.prompts)

    @patch("pyrit.datasets.harmbench_multimodal_dataset._fetch_and_save_image_async")
    @patch("pyrit.datasets.harmbench_multimodal_dataset.fetch_examples")
    async def test_filtering_by_semantic_categories(self, mock_fetch, mock_fetch_image):
        mock_examples = [
            {
                "Behavior": "Illegal behavior",
                "BehaviorID": "illegal_behavior",
                "SemanticCategory": "illegal",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "illegal.png",
            },
            {
                "Behavior": "Harmful behavior",
                "BehaviorID": "harmful_behavior",
                "SemanticCategory": "harmful",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "harmful.png",
            },
            {
                "Behavior": "Cybercrime behavior",
                "BehaviorID": "cybercrime_behavior",
                "SemanticCategory": "cybercrime_intrusion",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "cybercrime.png",
            },
            {
                "Behavior": "Harassment behavior",
                "BehaviorID": "harassment_behavior",
                "SemanticCategory": "harassment_bullying",
                "FunctionalCategory": "multimodal",
                "ImageFileName": "harassment.png",
            },
        ]
        mock_fetch.return_value = mock_examples

        mock_fetch_image.side_effect = [
            "/dbdata/seed-prompt-entries/images/harmbench_illegal_behavior.png",
            "/dbdata/seed-prompt-entries/images/harmbench_cybercrime_behavior.png",
            "/dbdata/seed-prompt-entries/images/harmbench_harmful_behavior.png",
        ]

        # Filter by single category
        result = await fetch_harmbench_multimodal_dataset_async(categories=["illegal"])
        assert isinstance(result, SeedPromptDataset)
        assert len(result.prompts) == 2  # 1 text + 1 image prompt for illegal category
        assert all(p.metadata["behavior_id"] == "illegal_behavior" for p in result.prompts)
        assert all(p.harm_categories == ["illegal"] for p in result.prompts)

        # Filter by multiple categories
        result = await fetch_harmbench_multimodal_dataset_async(categories=["cybercrime_intrusion", "harmful"])
        assert isinstance(result, SeedPromptDataset)
        assert len(result.prompts) == 4  # 2 examples × 2 prompts each
        behavior_ids = {p.metadata["behavior_id"] for p in result.prompts}
        assert behavior_ids == {"cybercrime_behavior", "harmful_behavior"}

        # Filter with invalid category
        with pytest.raises(ValueError, match="Invalid semantic categories"):
            await fetch_harmbench_multimodal_dataset_async(categories=["nonexistent_category", "illegal"])

        # Filter with an empty list
        with pytest.raises(ValueError, match="SeedPromptDataset cannot be empty"):
            await fetch_harmbench_multimodal_dataset_async(categories=[])
