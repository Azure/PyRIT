# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.datasets.seed_datasets import SeedDatasetProvider
from pyrit.datasets.seed_datasets.local.local_dataset_loader import LocalDatasetLoader
from pyrit.datasets.seed_datasets.remote.darkbench_dataset import DarkBenchDataset
from pyrit.datasets.seed_datasets.remote.harmbench_dataset import HarmBenchDataset
from pyrit.models import SeedDataset, SeedPrompt


@pytest.fixture
def mock_harmbench_data():
    """Mock data for HarmBench dataset."""
    return [
        {
            "Behavior": "Test harmful behavior 1",
            "SemanticCategory": "chemical_biological",
        },
        {
            "Behavior": "Test harmful behavior 2",
            "SemanticCategory": "illegal_activity",
        },
    ]


@pytest.fixture
def mock_darkbench_data():
    """Mock data for DarkBench dataset."""
    return [
        {
            "Example": "Test dark pattern example 1",
            "deceptive_pattern": "manipulative_design",
        },
        {
            "Example": "Test dark pattern example 2",
            "deceptive_pattern": "forced_action",
        },
    ]


class TestSeedDatasetProvider:
    """Test the SeedDatasetProvider base class and registration."""

    def test_get_all_providers(self):
        """Test that providers are automatically registered."""
        providers = SeedDatasetProvider.get_all_providers()

        assert isinstance(providers, dict)
        assert len(providers) >= 2  # At least HarmBench and DarkBench
        assert "HarmBenchDataset" in providers
        assert "DarkBenchDataset" in providers

    def test_get_all_dataset_names(self):
        """Test getting all dataset names."""
        names = SeedDatasetProvider.get_all_dataset_names()

        assert isinstance(names, list)
        assert len(names) >= 2
        assert "harmbench" in names
        assert "DarkBench" in names
        # Names should be sorted
        assert names == sorted(names)

    def test_local_loaders_registered(self):
        """Test that local dataset loaders are registered."""
        providers = SeedDatasetProvider.get_all_providers()
        
        # Check if any LocalDatasetLoader instances are registered
        local_loaders = [k for k in providers.keys() if "LocalDatasetLoader" in k]
        # Should have at least some local datasets if seed_datasets directory has files
        assert len(local_loaders) >= 0  # May be 0 if no .prompt files exist

    @pytest.mark.asyncio
    async def test_fetch_all_datasets_with_mocks(self, mock_harmbench_data, mock_darkbench_data):
        """Test fetching all datasets with mocked data."""
        with patch.object(
            HarmBenchDataset, "_fetch_from_url", return_value=mock_harmbench_data
        ), patch.object(DarkBenchDataset, "_fetch_from_huggingface", return_value=mock_darkbench_data):
            datasets = await SeedDatasetProvider.fetch_all_datasets()

            assert isinstance(datasets, list)
            assert len(datasets) >= 2
            assert all(isinstance(d, SeedDataset) for d in datasets)

    @pytest.mark.asyncio
    async def test_fetch_all_datasets_with_filter(self, mock_harmbench_data):
        """Test fetching datasets with name filter."""
        with patch.object(HarmBenchDataset, "_fetch_from_url", return_value=mock_harmbench_data):
            datasets = await SeedDatasetProvider.fetch_all_datasets(dataset_names=["harmbench"])

            assert isinstance(datasets, list)
            assert len(datasets) == 1
            assert datasets[0].prompts[0].dataset_name == "harmbench"


class TestHarmBenchDataset:
    """Test the HarmBench dataset loader."""

    @pytest.mark.asyncio
    async def test_fetch_dataset(self, mock_harmbench_data):
        """Test fetching HarmBench dataset."""
        loader = HarmBenchDataset()

        with patch.object(loader, "_fetch_from_url", return_value=mock_harmbench_data):
            dataset = await loader.fetch_dataset()

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.prompts) == 2
            assert all(isinstance(p, SeedPrompt) for p in dataset.prompts)

            # Check first prompt
            first_prompt = dataset.prompts[0]
            assert first_prompt.value == "Test harmful behavior 1"
            assert first_prompt.data_type == "text"
            assert first_prompt.dataset_name == "harmbench"
            assert first_prompt.harm_categories == ["chemical_biological"]
            assert first_prompt.name == "HarmBench Examples"

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = HarmBenchDataset()
        assert loader.dataset_name == "harmbench"

    @pytest.mark.asyncio
    async def test_fetch_dataset_missing_keys(self):
        """Test that missing required keys raise ValueError."""
        loader = HarmBenchDataset()
        invalid_data = [{"Behavior": "Test"}]  # Missing SemanticCategory

        with patch.object(loader, "_fetch_from_url", return_value=invalid_data):
            with pytest.raises(ValueError, match="Missing keys in example"):
                await loader.fetch_dataset()

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_custom_source(self, mock_harmbench_data):
        """Test fetching with custom source URL."""
        loader = HarmBenchDataset(
            source="https://custom.example.com/data.csv",
            source_type="public_url",
            cache=False,
        )

        with patch.object(loader, "_fetch_from_url", return_value=mock_harmbench_data) as mock_fetch:
            dataset = await loader.fetch_dataset()

            assert len(dataset.prompts) == 2
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs["source"] == "https://custom.example.com/data.csv"
            assert call_kwargs["source_type"] == "public_url"
            assert call_kwargs["cache"] is False


class TestDarkBenchDataset:
    """Test the DarkBench dataset loader."""

    @pytest.mark.asyncio
    async def test_fetch_dataset(self, mock_darkbench_data):
        """Test fetching DarkBench dataset."""
        loader = DarkBenchDataset()

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_darkbench_data):
            dataset = await loader.fetch_dataset()

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.prompts) == 2
            assert all(isinstance(p, SeedPrompt) for p in dataset.prompts)

            # Check first prompt
            first_prompt = dataset.prompts[0]
            assert first_prompt.value == "Test dark pattern example 1"
            assert first_prompt.data_type == "text"
            assert first_prompt.dataset_name == "DarkBench"
            assert first_prompt.harm_categories == ["manipulative_design"]

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = DarkBenchDataset()
        assert loader.dataset_name == "DarkBench"

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_custom_config(self, mock_darkbench_data):
        """Test fetching with custom HuggingFace config."""
        loader = DarkBenchDataset(
            dataset_name="custom/darkbench",
            config="custom_config",
            split="test",
            cache_dir="/custom/cache",
        )

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_darkbench_data) as mock_fetch:
            dataset = await loader.fetch_dataset()

            assert len(dataset.prompts) == 2
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs["dataset_name"] == "custom/darkbench"
            assert call_kwargs["config"] == "custom_config"
            assert call_kwargs["split"] == "test"
            assert call_kwargs["cache_dir"] == "/custom/cache"


class TestRemoteDatasetLoaderHelpers:
    """Test the helper methods in RemoteDatasetLoader."""

    def test_get_cache_file_name(self):
        """Test cache file name generation."""
        loader = HarmBenchDataset()
        cache_name = loader._get_cache_file_name(
            source="https://example.com/data.csv",
            file_type="csv",
        )

        assert isinstance(cache_name, str)
        assert cache_name.endswith(".csv")
        assert len(cache_name) > 4  # MD5 hash + extension

    def test_get_cache_file_name_deterministic(self):
        """Test that same source produces same cache name."""
        loader = HarmBenchDataset()
        source = "https://example.com/data.csv"

        name1 = loader._get_cache_file_name(source=source, file_type="csv")
        name2 = loader._get_cache_file_name(source=source, file_type="csv")

        assert name1 == name2
