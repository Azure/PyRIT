# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, patch

import pytest

from pyrit.datasets.seed_datasets.remote.jailbreakv_dataset import (
    _JailbreakVDataset,
)
from pyrit.models import SeedDataset


class TestJailbreakVDataset:
    """Unit tests for _JailbreakVDataset."""

    def test_dataset_name(self):
        dataset = _JailbreakVDataset()
        assert dataset.dataset_name == "jailbreakv_28k"

    def test_init_defaults(self):
        dataset = _JailbreakVDataset()
        assert dataset.source == "JailbreakV-28K/JailBreakV-28k"
        assert dataset.config == "JailBreakV_28K"
        assert dataset.split == "mini_JailBreakV_28K"

    def test_init_redteam_config_default_split(self):
        dataset = _JailbreakVDataset(config="RedTeam_2K")
        assert dataset.config == "RedTeam_2K"
        assert dataset.split == "RedTeam_2K"

    def test_init_custom_split(self):
        dataset = _JailbreakVDataset(split="JailBreakV_28K")
        assert dataset.split == "JailBreakV_28K"

    @pytest.mark.asyncio
    async def test_fetch_dataset_jailbreakv_config(self):
        mock_data = [
            {
                "jailbreak_query": "Test jailbreak prompt 1",
                "redteam_query": "Test redteam query 1",
                "policy": "Violence",
                "format": "text",
                "from": "source1",
            },
            {
                "jailbreak_query": "Test jailbreak prompt 2",
                "redteam_query": "Test redteam query 2",
                "policy": "Fraud",
                "format": "text",
                "from": "source2",
            },
        ]

        dataset_loader = _JailbreakVDataset()

        with patch.object(dataset_loader, "_fetch_from_huggingface", new_callable=AsyncMock, return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert isinstance(dataset, SeedDataset)
            assert dataset.dataset_name == "jailbreakv_28k"
            assert len(dataset.seeds) == 2
            assert dataset.seeds[0].value == "Test jailbreak prompt 1"
            assert dataset.seeds[0].harm_categories == ["Violence"]
            assert dataset.seeds[0].metadata["redteam_query"] == "Test redteam query 1"
            assert dataset.seeds[0].metadata["format"] == "text"
            assert dataset.seeds[0].metadata["from"] == "source1"
            assert dataset.seeds[1].harm_categories == ["Fraud"]

    @pytest.mark.asyncio
    async def test_fetch_dataset_redteam_config(self):
        mock_data = [
            {
                "question": "Test red team question",
                "policy": "Hate Speech",
            },
        ]

        dataset_loader = _JailbreakVDataset(config="RedTeam_2K")

        with patch.object(dataset_loader, "_fetch_from_huggingface", new_callable=AsyncMock, return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.seeds) == 1
            assert dataset.seeds[0].value == "Test red team question"
            assert dataset.seeds[0].harm_categories == ["Hate Speech"]

    @pytest.mark.asyncio
    async def test_fetch_dataset_skips_empty_prompts(self):
        mock_data = [
            {"jailbreak_query": "", "policy": "Violence"},
            {"jailbreak_query": "   ", "policy": "Fraud"},
            {"jailbreak_query": "Valid prompt", "policy": "Malware"},
        ]

        dataset_loader = _JailbreakVDataset()

        with patch.object(dataset_loader, "_fetch_from_huggingface", new_callable=AsyncMock, return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 1
            assert dataset.seeds[0].value == "Valid prompt"

    @pytest.mark.asyncio
    async def test_fetch_dataset_empty_raises_error(self):
        mock_data = [
            {"jailbreak_query": "", "policy": "Violence"},
        ]

        dataset_loader = _JailbreakVDataset()

        with patch.object(dataset_loader, "_fetch_from_huggingface", new_callable=AsyncMock, return_value=mock_data):
            with pytest.raises(Exception, match="Error loading JailBreakV-28K dataset"):
                await dataset_loader.fetch_dataset(cache=False)

    @pytest.mark.asyncio
    async def test_fetch_dataset_empty_policy(self):
        mock_data = [
            {"jailbreak_query": "Test prompt", "policy": ""},
        ]

        dataset_loader = _JailbreakVDataset()

        with patch.object(dataset_loader, "_fetch_from_huggingface", new_callable=AsyncMock, return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 1
            assert dataset.seeds[0].harm_categories == []

    @pytest.mark.asyncio
    async def test_fetch_dataset_optional_metadata_missing(self):
        mock_data = [
            {"jailbreak_query": "Test prompt", "policy": "Violence"},
        ]

        dataset_loader = _JailbreakVDataset()

        with patch.object(dataset_loader, "_fetch_from_huggingface", new_callable=AsyncMock, return_value=mock_data):
            dataset = await dataset_loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 1
            assert dataset.seeds[0].metadata == {}
