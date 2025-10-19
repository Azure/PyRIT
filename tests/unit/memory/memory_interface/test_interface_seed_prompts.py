# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from typing import Sequence
from uuid import uuid4

import pytest

from pyrit.memory import MemoryInterface
from pyrit.models import MessagePiece, SeedPrompt, SeedPromptGroup


def assert_original_value_in_list(original_value: str, message_pieces: Sequence[MessagePiece]):
    for piece in message_pieces:
        if piece.original_value == original_value:
            return True
    raise AssertionError(f"Original value {original_value} not found in list")


@pytest.mark.asyncio
async def test_get_seed_prompts_with_audio(sqlite_instance: MemoryInterface):
    """Test adding and retrieving seed prompts with an audio file."""
    temp_files = []
    try:
        # Create a temporary audio file
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_file.write(b"dummy audio content")
        audio_file.close()
        temp_files.append(audio_file.name)

        # Create a seed prompt for the audio file
        audio_prompt = SeedPrompt(value=audio_file.name, dataset_name="dataset_audio", data_type="audio_path")

        # Add seed prompt to memory
        await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[audio_prompt], added_by="test_audio")

        # Retrieve and verify the seed prompts
        result = sqlite_instance.get_seed_prompts()
        assert len(result) == 1
        assert result[0].value.endswith(".wav")
        assert result[0].data_type == "audio_path"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.asyncio
async def test_get_seed_prompts_with_video(sqlite_instance: MemoryInterface):
    """Test adding and retrieving seed prompts with a video file."""
    temp_files = []
    try:
        # Create a temporary video file
        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_file.write(b"dummy video content")
        video_file.close()
        temp_files.append(video_file.name)

        # Create a seed prompt for the video file
        video_prompt = SeedPrompt(value=video_file.name, dataset_name="dataset_video", data_type="video_path")

        # Add seed prompt to memory
        await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[video_prompt], added_by="test_video")

        # Retrieve and verify the seed prompts
        result = sqlite_instance.get_seed_prompts()
        assert len(result) == 1
        assert result[0].value.endswith(".mp4")
        assert result[0].data_type == "video_path"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.asyncio
async def test_get_seed_prompts_with_image(sqlite_instance: MemoryInterface):
    """Test adding and retrieving seed prompts with an image file."""
    temp_files = []
    try:
        # Create a temporary image file
        image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_file.write(b"dummy image content")
        image_file.close()
        temp_files.append(image_file.name)

        # Create a seed prompt for the image file
        image_prompt = SeedPrompt(value=image_file.name, dataset_name="dataset_image", data_type="image_path")

        # Add seed prompt to memory
        await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[image_prompt], added_by="test_image")

        # Retrieve and verify the seed prompts
        result = sqlite_instance.get_seed_prompts()
        assert len(result) == 1
        assert result[0].value.endswith(".png")
        assert result[0].data_type == "image_path"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.asyncio
async def test_get_seed_prompts_with_value_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="another prompt", dataset_name="dataset2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(value="prompt1")
    assert len(result) == 1
    assert result[0].value == "prompt1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_dataset_name_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(dataset_name="dataset1")
    assert len(result) == 1
    assert result[0].dataset_name == "dataset1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_added_by_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", added_by="user1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", added_by="user2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts)

    result = sqlite_instance.get_seed_prompts(added_by="user1")
    assert len(result) == 1
    assert result[0].added_by == "user1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_source_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", source="source1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", source="source2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(source="source1")
    assert len(result) == 1
    assert result[0].source == "source1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_harm_categories_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(harm_categories=["category1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_authors_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", authors=["author2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(authors=["author1"])
    assert len(result) == 1
    assert result[0].authors == ["author1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_groups_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", groups=["group1"], data_type="text"),
        SeedPrompt(value="prompt2", groups=["group2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(groups=["group1"])
    assert len(result) == 1
    assert result[0].groups == ["group1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_parameters_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", parameters=["param1"], data_type="text"),
        SeedPrompt(value="prompt2", parameters=["param2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(parameters=["param1"])
    assert len(result) == 1
    assert result[0].parameters == ["param1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_metadata_filter(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", data_type="text", metadata={"key1": "value1", "key2": "value2"}),
        SeedPrompt(value="prompt2", data_type="text", metadata={"key1": "value2"}),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(metadata={"key1": "value1"})
    assert len(result) == 1
    assert result[0].metadata == {"key1": "value1", "key2": "value2"}


@pytest.mark.asyncio
async def test_get_seed_prompts_with_multiple_filters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", added_by="user1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", added_by="user2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts)

    result = sqlite_instance.get_seed_prompts(dataset_name="dataset1", added_by="user1")
    assert len(result) == 1
    assert result[0].dataset_name == "dataset1"
    assert result[0].added_by == "user1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_empty_list_filters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["harm1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["harm2"], authors=["author2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(harm_categories=[], authors=[])
    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_seed_prompts_with_single_element_list_filters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], authors=["author2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(harm_categories=["category1"], authors=["author1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]
    assert result[0].authors == ["author1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_multiple_elements_list_filters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(
            value="prompt1",
            harm_categories=["category1", "category2"],
            authors=["author1", "author2"],
            data_type="text",
        ),
        SeedPrompt(value="prompt2", harm_categories=["category3"], authors=["author3"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(
        harm_categories=["category1", "category2"], authors=["author1", "author2"]
    )
    assert len(result) == 1
    assert result[0].harm_categories == ["category1", "category2"]
    assert result[0].authors == ["author1", "author2"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_multiple_elements_list_filters_additional(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(
            value="prompt1",
            harm_categories=["category1", "category2"],
            authors=["author1", "author2"],
            data_type="text",
        ),
        SeedPrompt(value="prompt2", harm_categories=["category3"], authors=["author3"], data_type="text"),
        SeedPrompt(
            value="prompt3",
            harm_categories=["category1", "category3"],
            authors=["author1", "author3"],
            data_type="text",
        ),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(
        harm_categories=["category1", "category3"], authors=["author1", "author3"]
    )
    assert len(result) == 1
    assert result[0].harm_categories == ["category1", "category3"]
    assert result[0].authors == ["author1", "author3"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_substring_filters_harm_categories(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], authors=["author2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(harm_categories=["ory1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]

    result = sqlite_instance.get_seed_prompts(authors=["auth"])
    assert len(result) == 2
    assert result[0].authors == ["author1"]
    assert result[1].authors == ["author2"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_substring_filters_groups(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", groups=["group1"], data_type="text"),
        SeedPrompt(value="prompt2", groups=["group2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(groups=["oup1"])
    assert len(result) == 1
    assert result[0].groups == ["group1"]

    result = sqlite_instance.get_seed_prompts(groups=["oup"])
    assert len(result) == 2
    assert result[0].groups == ["group1"]
    assert result[1].groups == ["group2"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_substring_filters_parameters(sqlite_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", parameters=["param1"], data_type="text"),
        SeedPrompt(value="prompt2", parameters=["param2"], data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = sqlite_instance.get_seed_prompts(parameters=["ram1"])
    assert len(result) == 1
    assert result[0].parameters == ["param1"]

    result = sqlite_instance.get_seed_prompts(parameters=["ram"])
    assert len(result) == 2
    assert result[0].parameters == ["param1"]
    assert result[1].parameters == ["param2"]


@pytest.mark.asyncio
async def test_add_seed_prompts_to_memory_empty_list(sqlite_instance: MemoryInterface):
    prompts: Sequence[SeedPrompt] = []
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=prompts, added_by="tester")
    stored_prompts = sqlite_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 0


@pytest.mark.asyncio
async def test_add_seed_prompts_duplicate_entries_same_dataset(sqlite_instance: MemoryInterface):
    prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="test_dataset", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=prompts, added_by="tester")
    stored_prompts = sqlite_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 2

    # Try to add prompt list with one duplicate prompt and one new prompt
    duplicate_prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset", data_type="text"),
        SeedPrompt(value="prompt3", dataset_name="test_dataset", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=duplicate_prompts, added_by="tester")

    # Validate that only new prompt is added and the total prompt count is 3
    stored_prompts = sqlite_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 3


@pytest.mark.asyncio
async def test_add_seed_prompts_duplicate_entries_different_datasets(sqlite_instance: MemoryInterface):
    prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="test_dataset", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=prompts, added_by="tester")
    stored_prompts = sqlite_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 2

    # Try to add prompt list with one duplicate prompt and one new prompt
    duplicate_prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset2", data_type="text"),
        SeedPrompt(value="prompt3", dataset_name="test_dataset2", data_type="text"),
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=duplicate_prompts, added_by="tester")

    # Validate that only new prompt is added and the total prompt count is 3
    stored_prompts = sqlite_instance.get_seed_prompts()
    assert len(stored_prompts) == 4


def test_get_seed_prompt_dataset_names_empty(sqlite_instance: MemoryInterface):
    assert sqlite_instance.get_seed_prompt_dataset_names() == []


@pytest.mark.asyncio
async def test_get_seed_prompt_dataset_names_single(sqlite_instance: MemoryInterface):
    dataset_name = "test_dataset"
    seed_prompt = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[seed_prompt])
    assert sqlite_instance.get_seed_prompt_dataset_names() == [dataset_name]


@pytest.mark.asyncio
async def test_get_seed_prompt_dataset_names_single_dataset_multiple_entries(sqlite_instance: MemoryInterface):
    dataset_name = "test_dataset"
    seed_prompt1 = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    seed_prompt2 = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[seed_prompt1, seed_prompt2])
    assert sqlite_instance.get_seed_prompt_dataset_names() == [dataset_name]


@pytest.mark.asyncio
async def test_get_seed_prompt_dataset_names_multiple(sqlite_instance: MemoryInterface):
    dataset_names = [f"dataset_{i}" for i in range(5)]
    seed_prompts = [
        SeedPrompt(value=f"value_{i}", dataset_name=dataset_name, added_by="tester", data_type="text")
        for i, dataset_name in enumerate(dataset_names)
    ]
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts)
    assert len(sqlite_instance.get_seed_prompt_dataset_names()) == 5
    assert sorted(sqlite_instance.get_seed_prompt_dataset_names()) == sorted(dataset_names)


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_empty_list(sqlite_instance: MemoryInterface):
    prompt_group = SeedPromptGroup(
        prompts=[SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)]
    )
    prompt_group.prompts = []
    with pytest.raises(ValueError, match="Prompt group must have at least one prompt."):
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_single_element(sqlite_instance: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group], added_by="tester")
    assert len(sqlite_instance.get_seed_prompts()) == 1


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_elements(sqlite_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0, role="user")
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1, role="user")
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group], added_by="tester")
    assert len(sqlite_instance.get_seed_prompts()) == 2
    assert len(sqlite_instance.get_seed_prompt_groups()) == 1


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_no_elements(sqlite_instance: MemoryInterface):
    with pytest.raises(ValueError, match="SeedPromptGroup cannot be empty."):
        prompt_group = SeedPromptGroup(prompts=[])
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_single_element_no_added_by(sqlite_instance: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    with pytest.raises(ValueError, match="The 'added_by' attribute must be set for each prompt."):
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_elements_no_added_by(sqlite_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", data_type="text", sequence=0, role="user")
    prompt2 = SeedPrompt(value="Test prompt 2", data_type="text", sequence=1, role="user")
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    with pytest.raises(ValueError, match="The 'added_by' attribute must be set for each prompt."):
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_inconsistent_group_ids(sqlite_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0, role="user")
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1, role="user")

    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    prompt_group.prompts[0].prompt_group_id = uuid4()

    with pytest.raises(ValueError):
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_single_element_with_added_by(sqlite_instance: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])
    assert len(sqlite_instance.get_seed_prompts()) == 1


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_elements_with_added_by(sqlite_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0, role="user")
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1, role="user")
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])
    assert len(sqlite_instance.get_seed_prompts()) == 2


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_groups_with_added_by(sqlite_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0, role="user")
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1, role="user")
    prompt3 = SeedPrompt(value="Test prompt 3", added_by="tester", data_type="text", sequence=0, role="user")
    prompt4 = SeedPrompt(value="Test prompt 4", added_by="tester", data_type="text", sequence=1, role="user")

    prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2])
    prompt_group2 = SeedPromptGroup(prompts=[prompt3, prompt4])

    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group1, prompt_group2])
    assert len(sqlite_instance.get_seed_prompts()) == 4
    groups_from_memory = sqlite_instance.get_seed_prompt_groups()
    assert len(groups_from_memory) == 2
    assert groups_from_memory[0].prompts[0].id != groups_from_memory[1].prompts[1].id
    assert groups_from_memory[0].prompts[0].prompt_group_id == groups_from_memory[0].prompts[1].prompt_group_id
    assert groups_from_memory[1].prompts[0].prompt_group_id == groups_from_memory[1].prompts[1].prompt_group_id


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_with_all_modalities(sqlite_instance: MemoryInterface):
    """Test adding multiple prompt groups with different modalities using temporary files."""
    temp_files = []
    temp_dir = tempfile.TemporaryDirectory()
    sqlite_instance.results_path = temp_dir.name
    try:
        # Create a temporary image file
        image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_file.write(b"dummy image content")
        image_file.close()
        temp_files.append(image_file.name)

        # Create a temporary audio file
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_file.write(b"dummy audio content")
        audio_file.close()
        temp_files.append(audio_file.name)

        # Create a temporary video file
        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_file.write(b"dummy video content")
        video_file.close()
        temp_files.append(video_file.name)

        # Create prompts with the temporary file paths
        prompt1 = SeedPrompt(
            value=image_file.name, added_by="testmultimodal", data_type="image_path", sequence=0, role="user"
        )
        prompt2 = SeedPrompt(
            value=audio_file.name, added_by="testmultimodal", data_type="audio_path", sequence=1, role="system"
        )
        prompt3 = SeedPrompt(
            value=video_file.name, added_by="testmultimodal", data_type="video_path", sequence=2, role="system"
        )
        prompt4 = SeedPrompt(
            value="Test prompt 4", added_by="testmultimodal", data_type="text", sequence=3, role="user"
        )

        # Create SeedPromptGroup
        seed_prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2, prompt3, prompt4])

        # Add prompt groups to memory
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[seed_prompt_group1])

        # Assert the total number of prompts in memory
        assert len(sqlite_instance.get_seed_prompts(added_by="testmultimodal")) == 4

        # Retrieve and verify prompt groups from memory
        groups_from_memory = sqlite_instance.get_seed_prompt_groups(added_by="testmultimodal")
        assert len(groups_from_memory) == 1

        # Verify prompt group IDs are consistent within each group
        expected_prompt_group_id = groups_from_memory[0].prompts[0].prompt_group_id
        assert groups_from_memory[0].prompts[0].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[1].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[2].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[3].prompt_group_id == expected_prompt_group_id

        # Verify the specific data types and values
        assert groups_from_memory[0].prompts[0].data_type == "image_path"
        assert groups_from_memory[0].prompts[0].value.endswith(".png")
        assert groups_from_memory[0].prompts[1].data_type == "audio_path"
        assert groups_from_memory[0].prompts[1].value.endswith(".wav")
        assert groups_from_memory[0].prompts[2].data_type == "video_path"
        assert groups_from_memory[0].prompts[2].value.endswith(".mp4")
        assert groups_from_memory[0].prompts[3].data_type == "text"
        assert groups_from_memory[0].prompts[3].value == "Test prompt 4"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        temp_dir.cleanup()


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_with_textimage_modalities(sqlite_instance: MemoryInterface):
    """Test adding multiple prompt groups with text and image modalities using temporary files."""
    temp_files = []
    temp_dir = tempfile.TemporaryDirectory()
    sqlite_instance.results_path = temp_dir.name
    try:
        # Create a temporary image file
        image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_file.write(b"dummy image content")
        image_file.close()
        temp_files.append(image_file.name)

        # Create prompts with the temporary file paths
        prompt1 = SeedPrompt(
            value=image_file.name, added_by="testtextimagemultimodal", data_type="image_path", sequence=0, role="user"
        )
        prompt2 = SeedPrompt(
            value="Test prompt 2", added_by="testtextimagemultimodal", data_type="text", sequence=3, role="user"
        )

        # Create SeedPromptGroup
        seed_prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2])

        # Add prompt groups to memory
        await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[seed_prompt_group1])

        # Assert the total number of prompts in memory
        assert len(sqlite_instance.get_seed_prompts(added_by="testtextimagemultimodal")) == 2

        # Retrieve and verify prompt groups from memory
        groups_from_memory = sqlite_instance.get_seed_prompt_groups(added_by="testtextimagemultimodal")
        assert len(groups_from_memory) == 1

        # Verify prompt group IDs are consistent within each group
        expected_prompt_group_id = groups_from_memory[0].prompts[0].prompt_group_id
        assert groups_from_memory[0].prompts[0].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[1].prompt_group_id == expected_prompt_group_id

        # Verify the specific data types and values
        assert groups_from_memory[0].prompts[0].data_type == "image_path"
        assert groups_from_memory[0].prompts[0].value.endswith(".png")
        assert groups_from_memory[0].prompts[1].data_type == "text"
        assert groups_from_memory[0].prompts[1].value == "Test prompt 2"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        temp_dir.cleanup()


@pytest.mark.asyncio
async def test_get_seed_prompts_with_param_filters(sqlite_instance: MemoryInterface):
    template_value = "Test template {{ param1 }}"
    dataset_name = "dataset_1"
    harm_categories = ["category1"]
    added_by = "tester"
    parameters = ["param1"]
    template = SeedPrompt(
        value=template_value,
        dataset_name=dataset_name,
        parameters=parameters,
        harm_categories=harm_categories,
        added_by=added_by,
        data_type="text",
    )
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[template])

    templates = sqlite_instance.get_seed_prompts(
        value=template_value,
        dataset_name=dataset_name,
        harm_categories=harm_categories,
        added_by=added_by,
        parameters=parameters,
    )
    assert len(templates) == 1
    assert templates[0].value == template_value


def test_get_seed_prompt_groups_empty(sqlite_instance: MemoryInterface):
    assert sqlite_instance.get_seed_prompt_groups() == []


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_with_dataset_name(sqlite_instance: MemoryInterface):
    dataset_name = "test_dataset"
    prompt_group = SeedPromptGroup(
        prompts=[
            SeedPrompt(value="Test prompt", dataset_name=dataset_name, added_by="tester", data_type="text", sequence=0)
        ]
    )
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])

    groups = sqlite_instance.get_seed_prompt_groups(dataset_name=dataset_name)
    assert len(groups) == 1
    assert groups[0].prompts[0].dataset_name == dataset_name


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_with_multiple_filters(sqlite_instance: MemoryInterface):
    dataset_name = "dataset_1"
    data_types = ["text"]
    harm_categories = ["category1"]
    added_by = "tester"
    group = SeedPromptGroup(
        prompts=[
            SeedPrompt(
                value="Test prompt",
                dataset_name=dataset_name,
                harm_categories=harm_categories,
                added_by=added_by,
                sequence=0,
                data_type="text",
            )
        ]
    )
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[group])

    groups = sqlite_instance.get_seed_prompt_groups(
        dataset_name=dataset_name,
        data_types=data_types,
        harm_categories=harm_categories,
        added_by=added_by,
    )
    assert len(groups) == 1
    assert groups[0].prompts[0].dataset_name == dataset_name
    assert groups[0].prompts[0].added_by == added_by


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_multiple_groups(sqlite_instance: MemoryInterface):
    group1 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 1", dataset_name="dataset_1", added_by="user1", sequence=0, data_type="text")]
    )
    group2 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 2", dataset_name="dataset_2", added_by="user2", sequence=0, data_type="text")]
    )
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[group1, group2])

    groups = sqlite_instance.get_seed_prompt_groups()
    assert len(groups) == 2


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_multiple_groups_with_unique_ids(sqlite_instance: MemoryInterface):
    group1 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 1", dataset_name="dataset_1", added_by="user1", sequence=0, data_type="text")]
    )
    group2 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 2", dataset_name="dataset_2", added_by="user2", sequence=0, data_type="text")]
    )
    await sqlite_instance.add_seed_prompt_groups_to_memory(prompt_groups=[group1, group2])

    groups = sqlite_instance.get_seed_prompt_groups()
    assert len(groups) == 2
    # Check that each group has a unique prompt_group_id
    assert groups[0].prompts[0].prompt_group_id != groups[1].prompts[0].prompt_group_id


@pytest.mark.asyncio
async def test_get_seed_prompts_by_hash(sqlite_instance: MemoryInterface):
    entries = [
        SeedPrompt(value="Hello 1", data_type="text"),
        SeedPrompt(value="Hello 2", data_type="text"),
    ]

    hello_1_hash = "724c531a3bc130eb46fbc4600064779552682ef4f351976fe75d876d94e8088c"

    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=entries, added_by="rlundeen")
    retrieved_entries = sqlite_instance.get_seed_prompts(value_sha256=[hello_1_hash])

    assert len(retrieved_entries) == 1
    assert retrieved_entries[0].value == "Hello 1"
    assert retrieved_entries[0].value_sha256 == hello_1_hash
