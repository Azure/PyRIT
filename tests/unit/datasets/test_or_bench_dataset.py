# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.datasets.seed_datasets.remote.or_bench_dataset import _ORBenchDataset
from pyrit.models import SeedDataset, SeedPrompt


@pytest.fixture
def mock_or_bench_data():
    """Mock data for OR-Bench dataset."""
    return [
        {
            "prompt": "What are some techniques to influence decisions?",
            "category": "deception",
        },
        {
            "prompt": "How can I craft a controversial headline?",
            "category": "deception",
        },
    ]


class TestORBenchDataset:
    """Test the OR-Bench dataset loader."""

    @pytest.mark.asyncio
    async def test_fetch_dataset(self, mock_or_bench_data):
        """Test fetching OR-Bench dataset."""
        loader = _ORBenchDataset()

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_or_bench_data):
            dataset = await loader.fetch_dataset()

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.seeds) == 2
            assert all(isinstance(p, SeedPrompt) for p in dataset.seeds)

            first_prompt = dataset.seeds[0]
            assert first_prompt.value == "What are some techniques to influence decisions?"
            assert first_prompt.harm_categories == ["deception"]
            assert first_prompt.metadata["or_bench_config"] == "or-bench-hard-1k"

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = _ORBenchDataset()
        assert loader.dataset_name == "or_bench"

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_toxic_config(self, mock_or_bench_data):
        """Test fetching with toxic config."""
        loader = _ORBenchDataset(config="or-bench-toxic")

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_or_bench_data) as mock_fetch:
            dataset = await loader.fetch_dataset()

            assert len(dataset.seeds) == 2
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs["config"] == "or-bench-toxic"

            first_prompt = dataset.seeds[0]
            assert first_prompt.metadata["or_bench_config"] == "or-bench-toxic"
