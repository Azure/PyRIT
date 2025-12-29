# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.datasets import SeedDatasetProvider
from pyrit.datasets.seed_datasets.remote.darkbench_dataset import _DarkBenchDataset
from pyrit.datasets.seed_datasets.remote.harmbench_dataset import _HarmBenchDataset
from pyrit.datasets.seed_datasets.remote.jailbreakv_28k_dataset import (
    HarmCategory,
    _JailbreakV28KDataset,
)
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


@pytest.fixture
def mock_jailbreakv_data():
    """Mock data for JailbreakV-28K dataset."""
    return [
        {
            "jailbreak_query": "Test jailbreak query 1",
            "redteam_query": "Test redteam query 1",
            "policy": "Hate Speech",
            "image_path": "images/test_001.png",
        },
        {
            "jailbreak_query": "Test jailbreak query 2",
            "redteam_query": "Test redteam query 2",
            "policy": "Violence",
            "image_path": "images/test_002.png",
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
                return SeedDataset(seeds=[])

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
    async def test_fetch_datasets_async(self):
        """Test fetching all datasets."""
        # Mock providers
        mock_provider1 = MagicMock()
        mock_provider1.return_value.dataset_name = "d1"
        mock_provider1.return_value.fetch_dataset = AsyncMock(
            return_value=SeedDataset(seeds=[SeedPrompt(value="p1", data_type="text")], dataset_name="d1")
        )

        mock_provider2 = MagicMock()
        mock_provider2.return_value.dataset_name = "d2"
        mock_provider2.return_value.fetch_dataset = AsyncMock(
            return_value=SeedDataset(seeds=[SeedPrompt(value="p2", data_type="text")], dataset_name="d2")
        )

        with patch.dict(SeedDatasetProvider._registry, {"P1": mock_provider1, "P2": mock_provider2}, clear=True):
            datasets = await SeedDatasetProvider.fetch_datasets_async()
            assert len(datasets) == 2

    @pytest.mark.asyncio
    async def test_fetch_datasets_async_with_filter(self):
        """Test fetching datasets with filter."""
        mock_provider1 = MagicMock()
        mock_provider1.return_value.dataset_name = "d1"
        mock_provider1.return_value.fetch_dataset = AsyncMock(
            return_value=SeedDataset(seeds=[SeedPrompt(value="p1", data_type="text")], dataset_name="d1")
        )

        mock_provider2 = MagicMock()
        mock_provider2.return_value.dataset_name = "d2"
        mock_provider2.return_value.fetch_dataset = AsyncMock(side_effect=Exception("Should not be called"))

        with patch.dict(SeedDatasetProvider._registry, {"P1": mock_provider1, "P2": mock_provider2}, clear=True):
            datasets = await SeedDatasetProvider.fetch_datasets_async(dataset_names=["d1"])
            assert len(datasets) == 1
            assert datasets[0].dataset_name == "d1"

    @pytest.mark.asyncio
    async def test_fetch_datasets_async_invalid_dataset_name(self):
        """Test that fetch_datasets_async raises ValueError for invalid dataset names."""
        mock_provider1 = MagicMock()
        mock_provider1.return_value.dataset_name = "d1"
        mock_provider1.return_value.fetch_dataset = AsyncMock(
            return_value=SeedDataset(seeds=[SeedPrompt(value="p1", data_type="text")], dataset_name="d1")
        )

        mock_provider2 = MagicMock()
        mock_provider2.return_value.dataset_name = "d2"
        mock_provider2.return_value.fetch_dataset = AsyncMock(
            return_value=SeedDataset(seeds=[SeedPrompt(value="p2", data_type="text")], dataset_name="d2")
        )

        with patch.dict(SeedDatasetProvider._registry, {"P1": mock_provider1, "P2": mock_provider2}, clear=True):
            # Test with single invalid name
            with pytest.raises(ValueError, match=r"Dataset\(s\) not found: \['nonexistent'\]"):
                await SeedDatasetProvider.fetch_datasets_async(dataset_names=["nonexistent"])

            # Test with mix of valid and invalid names
            with pytest.raises(ValueError, match=r"Dataset\(s\) not found: \['invalid1', 'invalid2'\]"):
                await SeedDatasetProvider.fetch_datasets_async(dataset_names=["d1", "invalid1", "invalid2"])


class TestHarmBenchDataset:
    """Test the HarmBench dataset loader."""

    @pytest.mark.asyncio
    async def test_fetch_dataset(self, mock_harmbench_data):
        """Test fetching HarmBench dataset."""
        loader = _HarmBenchDataset()

        with patch.object(loader, "_fetch_from_url", return_value=mock_harmbench_data):
            dataset = await loader.fetch_dataset()

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.seeds) == 2
            assert all(isinstance(p, SeedPrompt) for p in dataset.seeds)

            # Check first prompt
            first_prompt = dataset.seeds[0]
            assert first_prompt.value == "Test harmful behavior 1"
            assert first_prompt.data_type == "text"
            assert first_prompt.dataset_name == "harmbench"
            assert first_prompt.harm_categories == ["chemical_biological"]
            assert first_prompt.name == "HarmBench Examples"

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = _HarmBenchDataset()
        assert loader.dataset_name == "harmbench"

    @pytest.mark.asyncio
    async def test_fetch_dataset_missing_keys(self):
        """Test that missing required keys raise ValueError."""
        loader = _HarmBenchDataset()
        invalid_data = [{"Behavior": "Test"}]  # Missing SemanticCategory

        with patch.object(loader, "_fetch_from_url", return_value=invalid_data):
            with pytest.raises(ValueError, match="Missing keys in example"):
                await loader.fetch_dataset()

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_custom_source(self, mock_harmbench_data):
        """Test fetching with custom source URL."""
        loader = _HarmBenchDataset(
            source="https://custom.example.com/data.csv",
            source_type="public_url",
        )

        with patch.object(loader, "_fetch_from_url", return_value=mock_harmbench_data) as mock_fetch:
            dataset = await loader.fetch_dataset(cache=False)

            assert len(dataset.seeds) == 2
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
        loader = _DarkBenchDataset()

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_darkbench_data):
            dataset = await loader.fetch_dataset()

            assert isinstance(dataset, SeedDataset)
            assert len(dataset.seeds) == 2
            assert all(isinstance(p, SeedPrompt) for p in dataset.seeds)

            # Check first prompt
            first_prompt = dataset.seeds[0]
            assert first_prompt.value == "Test dark pattern example 1"
            assert first_prompt.data_type == "text"
            assert first_prompt.dataset_name == "dark_bench"
            assert first_prompt.harm_categories == ["manipulative_design"]

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = _DarkBenchDataset()
        assert loader.dataset_name == "dark_bench"

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_custom_config(self, mock_darkbench_data):
        """Test fetching with custom HuggingFace config."""
        loader = _DarkBenchDataset(
            dataset_name="custom/darkbench",
            config="custom_config",
            split="test",
        )

        with patch.object(loader, "_fetch_from_huggingface", return_value=mock_darkbench_data) as mock_fetch:
            dataset = await loader.fetch_dataset()

            assert len(dataset.seeds) == 2
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs["dataset_name"] == "custom/darkbench"
            assert call_kwargs["config"] == "custom_config"
            assert call_kwargs["split"] == "test"


class TestJailbreakV28KDataset:
    """Test the JailbreakV-28K dataset loader."""

    @pytest.mark.asyncio
    async def test_fetch_dataset(self, mock_jailbreakv_data, tmp_path):
        """Test fetching JailbreakV-28K dataset."""
        # Create mock ZIP structure
        zip_dir = tmp_path / "test_zip"
        zip_dir.mkdir()
        images_dir = zip_dir / "JailBreakV_28K" / "images"
        images_dir.mkdir(parents=True)

        # Create mock image files
        (images_dir / "test_001.png").touch()
        (images_dir / "test_002.png").touch()

        loader = _JailbreakV28KDataset(
            zip_dir=str(zip_dir),
        )

        # Mock the ZIP extraction check
        with patch("pathlib.Path.exists") as mock_exists:
            # ZIP exists, extracted folder exists
            mock_exists.side_effect = lambda: True

            with patch.object(loader, "_fetch_from_huggingface", return_value=mock_jailbreakv_data):
                with patch.object(loader, "_resolve_image_path") as mock_resolve:
                    # Mock image path resolution
                    mock_resolve.side_effect = lambda rel_path, local_directory, call_cache: str(
                        local_directory / rel_path
                    )

                    dataset = await loader.fetch_dataset()

                    assert isinstance(dataset, SeedDataset)
                    # 2 examples * 2 prompts each (text + image) = 4 total
                    assert len(dataset.seeds) == 4
                    assert all(isinstance(p, SeedPrompt) for p in dataset.seeds)

                    # Check text prompts
                    text_prompts = [p for p in dataset.seeds if p.data_type == "text"]
                    assert len(text_prompts) == 2
                    assert text_prompts[0].value == "Test redteam query 1"
                    assert text_prompts[0].dataset_name == "jailbreakv_28k"
                    assert text_prompts[0].harm_categories == ["hate_speech"]

                    # Check image prompts
                    image_prompts = [p for p in dataset.seeds if p.data_type == "image_path"]
                    assert len(image_prompts) == 2

    def test_dataset_name(self):
        """Test dataset_name property."""
        loader = _JailbreakV28KDataset()
        assert loader.dataset_name == "jailbreakv_28k"

    def test_harm_category_enum(self):
        """Test HarmCategory enum values."""
        assert HarmCategory.HATE_SPEECH.value == "Hate Speech"
        assert HarmCategory.VIOLENCE.value == "Violence"
        assert HarmCategory.FRAUD.value == "Fraud"

    def test_initialization_with_harm_categories(self):
        """Test initialization with harm category filtering."""
        loader = _JailbreakV28KDataset(
            harm_categories=[HarmCategory.HATE_SPEECH, HarmCategory.VIOLENCE],
        )
        assert loader.harm_categories is not None
        assert len(loader.harm_categories) == 2
        assert HarmCategory.HATE_SPEECH in loader.harm_categories

    def test_initialization_invalid_harm_category(self):
        """Test that invalid harm categories raise ValueError."""
        with pytest.raises(ValueError, match="Invalid harm categories"):
            _JailbreakV28KDataset(
                harm_categories=["invalid_category"],  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_fetch_dataset_with_text_field(self, mock_jailbreakv_data, tmp_path):
        """Test fetching with different text field."""
        zip_dir = tmp_path / "test_zip"
        zip_dir.mkdir()
        images_dir = zip_dir / "JailBreakV_28K" / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "test_001.png").touch()
        (images_dir / "test_002.png").touch()

        loader = _JailbreakV28KDataset(
            zip_dir=str(zip_dir),
            text_field="jailbreak_query",
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(loader, "_fetch_from_huggingface", return_value=mock_jailbreakv_data):
                with patch.object(loader, "_resolve_image_path") as mock_resolve:
                    mock_resolve.side_effect = lambda rel_path, local_directory, call_cache: str(
                        local_directory / rel_path
                    )

                    dataset = await loader.fetch_dataset()

                    text_prompts = [p for p in dataset.seeds if p.data_type == "text"]
                    assert text_prompts[0].value == "Test jailbreak query 1"

    @pytest.mark.asyncio
    async def test_fetch_dataset_missing_zip(self):
        """Test that missing ZIP file raises FileNotFoundError."""
        loader = _JailbreakV28KDataset(
            zip_dir="/nonexistent/path",
        )

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="ZIP file not found"):
                await loader.fetch_dataset()

    @pytest.mark.asyncio
    async def test_fetch_dataset_filters_by_category(self, mock_jailbreakv_data, tmp_path):
        """Test filtering by harm categories."""
        zip_dir = tmp_path / "test_zip"
        zip_dir.mkdir()
        images_dir = zip_dir / "JailBreakV_28K" / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "test_001.png").touch()

        loader = _JailbreakV28KDataset(
            zip_dir=str(zip_dir),
            harm_categories=[HarmCategory.HATE_SPEECH],  # Only hate speech
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(loader, "_fetch_from_huggingface", return_value=mock_jailbreakv_data):
                with patch.object(loader, "_resolve_image_path") as mock_resolve:
                    mock_resolve.side_effect = lambda rel_path, local_directory, call_cache: (
                        str(local_directory / rel_path) if "test_001" in rel_path else ""
                    )

                    dataset = await loader.fetch_dataset()

                    # Should only get first example (hate speech), not second (violence)
                    text_prompts = [p for p in dataset.seeds if p.data_type == "text"]
                    assert len(text_prompts) == 1
                    assert text_prompts[0].harm_categories == ["hate_speech"]

    def test_normalize_policy(self):
        """Test policy normalization helper."""
        loader = _JailbreakV28KDataset()

        assert loader._normalize_policy("Hate Speech") == "hate_speech"
        assert loader._normalize_policy("Economic-Harm") == "economic_harm"
        assert loader._normalize_policy("  Violence  ") == "violence"

    @pytest.mark.asyncio
    async def test_fetch_dataset_50_percent_threshold(self, tmp_path):
        """Test that 50% or more missing images raises ValueError."""
        zip_dir = tmp_path / "test_zip"
        zip_dir.mkdir()
        images_dir = zip_dir / "JailBreakV_28K" / "images"
        images_dir.mkdir(parents=True)

        # Mock data with 4 items
        mock_data = [
            {"policy": "Hate Speech", "image_path": "images/001.png", "redteam_query": "Query 1"},
            {"policy": "Violence", "image_path": "images/002.png", "redteam_query": "Query 2"},
            {"policy": "Fraud", "image_path": "images/003.png", "redteam_query": "Query 3"},
            {"policy": "Malware", "image_path": "images/004.png", "redteam_query": "Query 4"},
        ]

        loader = _JailbreakV28KDataset(zip_dir=str(zip_dir))

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(loader, "_fetch_from_huggingface", return_value=mock_data):
                with patch.object(loader, "_resolve_image_path") as mock_resolve:
                    # Mock so that only 1 out of 4 images resolves (25% success = 75% unpaired)
                    def resolve_side_effect(rel_path, local_directory, call_cache):
                        return str(local_directory / rel_path) if "001" in rel_path else ""

                    mock_resolve.side_effect = resolve_side_effect

                    # Should raise because 75% are unpaired (>= 50%)
                    with pytest.raises(ValueError, match="75.0% of items are missing images"):
                        await loader.fetch_dataset()

    @pytest.mark.asyncio
    async def test_fetch_dataset_below_50_percent_threshold(self, tmp_path):
        """Test that less than 50% missing images succeeds."""
        zip_dir = tmp_path / "test_zip"
        zip_dir.mkdir()
        images_dir = zip_dir / "JailBreakV_28K" / "images"
        images_dir.mkdir(parents=True)

        # Mock data with 4 items
        mock_data = [
            {"policy": "Hate Speech", "image_path": "images/001.png", "redteam_query": "Query 1"},
            {"policy": "Violence", "image_path": "images/002.png", "redteam_query": "Query 2"},
            {"policy": "Fraud", "image_path": "images/003.png", "redteam_query": "Query 3"},
            {"policy": "Malware", "image_path": "images/004.png", "redteam_query": "Query 4"},
        ]

        loader = _JailbreakV28KDataset(zip_dir=str(zip_dir))

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(loader, "_fetch_from_huggingface", return_value=mock_data):
                with patch.object(loader, "_resolve_image_path") as mock_resolve:
                    # Mock so that 3 out of 4 images resolve (75% success = 25% unpaired)
                    def resolve_side_effect(rel_path, local_directory, call_cache):
                        return "" if "004" in rel_path else str(local_directory / rel_path)

                    mock_resolve.side_effect = resolve_side_effect

                    # Should succeed because only 25% are unpaired (< 50%)
                    dataset = await loader.fetch_dataset()

                    # Should have 3 pairs (6 total prompts)
                    assert len(dataset.seeds) == 6
