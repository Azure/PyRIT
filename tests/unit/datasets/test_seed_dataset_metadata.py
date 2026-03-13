# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for metadata components related to SeedDatasetProvider.
"""

from pyrit.datasets.seed_datasets.seed_metadata import (
    SeedDatasetFilter,
    SeedDatasetLoadingRank,
    SeedDatasetMetadata,
    SeedDatasetModality,
    SeedDatasetSize,
    SeedDatasetSourceType,
)


class TestMetadataLifecycle:
    """
    Test that the metadata object can be created with different
    subsets of values.
    """

    def test_has_no_values(self):
        metadata = SeedDatasetMetadata()
        assert metadata.tags is None
        assert metadata.size is None
        assert metadata.modalities is None
        assert metadata.source is None
        assert metadata.rank is None
        assert metadata.harm_categories is None

    def test_has_some_values(self):
        metadata = SeedDatasetMetadata(tags={"safety"}, size=SeedDatasetSize.LARGE)
        assert metadata.tags == {"safety"}
        assert metadata.size == SeedDatasetSize.LARGE
        assert metadata.modalities is None
        assert metadata.source is None
        assert metadata.rank is None
        assert metadata.harm_categories is None

    def test_has_all_values(self):
        metadata = SeedDatasetMetadata(
            tags={"default", "safety"},
            size=SeedDatasetSize.MEDIUM,
            modalities=[SeedDatasetModality.TEXT, SeedDatasetModality.IMAGE],
            source=SeedDatasetSourceType.REMOTE,
            rank=SeedDatasetLoadingRank.DEFAULT,
            harm_categories=["violence", "illegal"],
        )
        assert metadata.tags == {"default", "safety"}
        assert metadata.size == SeedDatasetSize.MEDIUM
        assert len(metadata.modalities) == 2
        assert metadata.source == SeedDatasetSourceType.REMOTE
        assert metadata.rank == SeedDatasetLoadingRank.DEFAULT
        assert metadata.harm_categories == ["violence", "illegal"]


class TestFilterLifecycle:
    """
    Test that the filter object can be created with different
    subsets of values.
    """

    def test_has_no_values(self):
        f = SeedDatasetFilter()
        assert f.tags is None
        assert f.sizes is None
        assert f.modalities is None
        assert f.sources is None
        assert f.ranks is None
        assert f.harm_categories is None

    def test_has_some_values(self):
        f = SeedDatasetFilter(sizes=[SeedDatasetSize.LARGE])
        assert f.sizes == [SeedDatasetSize.LARGE]
        assert f.tags is None
        assert f.modalities is None

    def test_has_all_values(self):
        f = SeedDatasetFilter(
            tags={"default"},
            sizes=[SeedDatasetSize.SMALL, SeedDatasetSize.MEDIUM],
            modalities=[SeedDatasetModality.TEXT],
            sources=[SeedDatasetSourceType.REMOTE],
            ranks=[SeedDatasetLoadingRank.DEFAULT],
            harm_categories=["violence"],
        )
        assert f.tags == {"default"}
        assert len(f.sizes) == 2
        assert f.modalities == [SeedDatasetModality.TEXT]
        assert f.sources == [SeedDatasetSourceType.REMOTE]
        assert f.ranks == [SeedDatasetLoadingRank.DEFAULT]
        assert f.harm_categories == ["violence"]


class TestMetadataProperties:
    """
    Test that the metadata fields populate correctly.
    """

    def test_size_value(self):
        for size in SeedDatasetSize:
            metadata = SeedDatasetMetadata(size=size)
            assert metadata.size == size

    def test_loading_rank_value(self):
        for rank in SeedDatasetLoadingRank:
            metadata = SeedDatasetMetadata(rank=rank)
            assert metadata.rank == rank

    def test_source_value(self):
        for source in SeedDatasetSourceType:
            metadata = SeedDatasetMetadata(source=source)
            assert metadata.source == source

    def test_modality_value(self):
        for modality in SeedDatasetModality:
            metadata = SeedDatasetMetadata(modalities=[modality])
            assert modality in metadata.modalities

    def test_tags_value(self):
        metadata = SeedDatasetMetadata(tags={"safety", "default", "custom"})
        assert "safety" in metadata.tags
        assert "default" in metadata.tags
        assert "custom" in metadata.tags

    def test_harm_categories_value(self):
        metadata = SeedDatasetMetadata(harm_categories=["violence", "cybercrime"])
        assert "violence" in metadata.harm_categories
        assert "cybercrime" in metadata.harm_categories


class TestFilterProperties:
    """
    Test that the filter fields populate correctly.
    """

    def test_sizes_values(self):
        f = SeedDatasetFilter(sizes=[SeedDatasetSize.SMALL, SeedDatasetSize.LARGE])
        assert SeedDatasetSize.SMALL in f.sizes
        assert SeedDatasetSize.LARGE in f.sizes

    def test_loading_ranks_values(self):
        f = SeedDatasetFilter(ranks=[SeedDatasetLoadingRank.DEFAULT, SeedDatasetLoadingRank.SLOW])
        assert SeedDatasetLoadingRank.DEFAULT in f.ranks
        assert SeedDatasetLoadingRank.SLOW in f.ranks

    def test_sources_values(self):
        f = SeedDatasetFilter(sources=[SeedDatasetSourceType.LOCAL, SeedDatasetSourceType.REMOTE])
        assert SeedDatasetSourceType.LOCAL in f.sources
        assert SeedDatasetSourceType.REMOTE in f.sources

    def test_modalities_values(self):
        f = SeedDatasetFilter(modalities=[SeedDatasetModality.TEXT, SeedDatasetModality.IMAGE])
        assert SeedDatasetModality.TEXT in f.modalities
        assert SeedDatasetModality.IMAGE in f.modalities

    def test_tags_values(self):
        f = SeedDatasetFilter(tags={"safety", "default"})
        assert "safety" in f.tags
        assert "default" in f.tags

    def test_harm_categories_values(self):
        f = SeedDatasetFilter(harm_categories=["violence", "cybercrime"])
        assert "violence" in f.harm_categories
        assert "cybercrime" in f.harm_categories
