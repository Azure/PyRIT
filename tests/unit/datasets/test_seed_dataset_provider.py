# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.datasets import SeedDatasetProvider
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
            "Deceptive Pattern": "manipulative_design",
        },
        {
            "Example": "Test dark pattern example 2",
            "Deceptive Pattern": "forced_action",
        },
    ]


class TestSeedDatasetProvider:
    """Test the SeedDatasetProvider base class and registration."""

    def test_registration(self):
        """Test that subclasses are automatically registered."""
        # Define a dynamic class to avoid polluting registry permanently (though it will stay)
        class DynamicTestProvider(SeedDatasetProvider):
            @property
            def dataset_name(self):
                return "dynamic_test"
            async def fetch_dataset(self):
                return SeedDataset(prompts=[])
        
        providers = SeedDatasetProvider.get_all_providers()
        assert "DynamicTestProvider" in providers
        assert providers["DynamicTestProvider"] == DynamicTestProvider

    def test_get_all_dataset_names(self):
        """Test getting all dataset names."""
        # Mock the registry to ensure deterministic results
        mock_provider_cls = MagicMock()
        mock_provider_instance = mock_provider_cls.return_value
        mock_provider_instance.dataset_name = "test_dataset"
        
        with patch.dict(SeedDatasetProvider._registry, {"TestProvider": mock_provider_cls}, clear=True):
            names = SeedDatasetProvider.get_all_dataset_names()
            assert names == ["test_dataset"]

    @pytest.mark.asyncio
    async def test_fetch_all_datasets(self):
        """Test fetching all datasets."""
        # Mock providers
        mock_provider1 = MagicMock()
        mock_provider1.return_value.dataset_name = "d1"
        mock_provider1.return_value.fetch_dataset = AsyncMock(return_value=SeedDataset(prompts=[SeedPrompt(value="p1", data_type="text")], dataset_name="d1"))
        
        mock_provider2 = MagicMock()
        mock_provider2.return_value.dataset_name = "d2"
        mock_provider2.return_value.fetch_dataset = AsyncMock(return_value=SeedDataset(prompts=[SeedPrompt(value="p2", data_type="text")], dataset_name="d2"))

        with patch.dict(SeedDatasetProvider._registry, {"P1": mock_provider1, "P2": mock_provider2}, clear=True):
            datasets = await SeedDatasetProvider.fetch_all_datasets()
            assert len(datasets) == 2
            
    @pytest.mark.asyncio
    async def test_fetch_all_datasets_with_filter(self):
        """Test fetching datasets with filter."""
        mock_provider1 = MagicMock()
        mock_provider1.return_value.dataset_name = "d1"
        mock_provider1.return_value.fetch_dataset = AsyncMock(return_value=SeedDataset(prompts=[SeedPrompt(value="p1", data_type="text")], dataset_name="d1"))
        
        mock_provider2 = MagicMock()
        mock_provider2.return_value.dataset_name = "d2"
        mock_provider2.return_value.fetch_dataset = AsyncMock(side_effect=Exception("Should not be called"))

        with patch.dict(SeedDatasetProvider._registry, {"P1": mock_provider1, "P2": mock_provider2}, clear=True):
            datasets = await SeedDatasetProvider.fetch_all_datasets(dataset_names=["d1"])
            assert len(datasets) == 1
            assert datasets[0].dataset_name == "d1"


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
        )

        with patch.object(loader, "_fetch_from_url", return_value=mock_harmbench_data) as mock_fetch:
            dataset = await loader.fetch_dataset(cache=False)

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
            assert first_prompt.dataset_name == "dark_bench"
            assert first_prompt.harm_categories == ["manipulative_design"]

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = DarkBenchDataset()
        assert loader.dataset_name == "dark_bench"

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_custom_config(self, mock_darkbench_data):
        """Test fetching with custom HuggingFace config."""
        loader = DarkBenchDataset(
            dataset_name="custom/darkbench",
            config="custom_config",
            split="test",
        )

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_darkbench_data) as mock_fetch:
            dataset = await loader.fetch_dataset()

            assert len(dataset.prompts) == 2
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs["dataset_name"] == "custom/darkbench"
            assert call_kwargs["config"] == "custom_config"
            assert call_kwargs["split"] == "test"



