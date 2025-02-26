# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 8. Seed Prompt Database
#
# Apart from storing results in memory it's also useful to store datasets of seed prompts
# and seed prompt templates that we may want to use at a later point.
# This can help us in curating prompts with custom metadata like harm categories.
# As with all memory, we can use local DuckDBMemory or AzureSQLMemory in Azure to get the
# benefits of sharing with other users and persisting data.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# %% [markdown]
# ## Adding prompts to the database

# %%
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.memory import CentralMemory
from pyrit.models import SeedPromptDataset

seed_prompt_dataset = SeedPromptDataset.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal-multimodal-dataset.prompt"
)

print(seed_prompt_dataset.prompts[0])

# Render user-defined values for yaml template
seed_prompt_dataset.render_template_value(stolen_item="a car")

memory = CentralMemory.get_memory_instance()
await memory.add_seed_prompts_to_memory_async(prompts=seed_prompt_dataset.prompts, added_by="test")  # type: ignore

# %% [markdown]
# ## Retrieving prompts from the database
#
# First, let's get an idea of what datasets are represented in the database.

# %%
memory.get_seed_prompt_dataset_names()

# %% [markdown]
# The dataset we just uploaded (called "test illegal") is also represented.
# To get all seed prompts from that dataset, we can query as follows:

# %%
dataset_name = "test illegal"
prompts = memory.get_seed_prompts(dataset_name=dataset_name)
print(f"Total number of the prompts with dataset name '{dataset_name}':", len(prompts))
for prompt in prompts:
    print(prompt.__dict__)

# %% [markdown]
# ## Adding multimodal seed prompt groups to the database
# In the following example, we will add a seed prompt group containing text, image, audio, and video prompts.
# When we add non-text seed prompts to memory, encoding data will automatically populate in the seed prompt's
# `metadata` field, including `format` (i.e. png, mp4, wav, etc.) as well as additional metadata for audio
# and video files, inclduing `bitrate` (kBits/s as int), `samplerate` (samples/second as int), `bitdepth` (as int),
# `filesize` (bytes as int), and `duration` (seconds as int) if the file type is supported by TinyTag.
# Example suppported file types include: MP3, MP4, M4A, and WAV. These may be helpful to filter for as some targets
# have specific input prompt requirements.
# %%
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPromptGroup

seed_prompt_group = SeedPromptGroup.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal-multimodal-group.prompt"
)

# Render user-defined values for yaml template
seed_prompt_group.render_template_value(stolen_item="a car")

await memory.add_seed_prompt_groups_to_memory(prompt_groups=[seed_prompt_group], added_by="test multimodal illegal")  # type: ignore

# %% [markdown]
# ## Retrieving seed prompt groups from the memory with dataset_name as "TestMultimodalTextImageAudioVideo"

# %%
multimodal_dataset_name = "TestMultimodalTextImageAudioVideo"
seed_prompt_groups = memory.get_seed_prompt_groups(dataset_name=multimodal_dataset_name)
print(f"Total number of the seed prompt groups with dataset name '{multimodal_dataset_name}':", len(seed_prompt_groups))
# Retrieving the auto-populated metadata for each seed prompt in the multimodal seed prompt group.
for seed_prompt in seed_prompt_group.prompts:
    print(f"SeedPrompt value: {seed_prompt.value}, SeedPrompt metadata: {seed_prompt.metadata}")

# %% [markdown]
# ## Filtering seed prompts by metadata
# %%
# Filter by metadata to get seed prompts in .wav format and sample rate 24000 kBits/s
memory.get_seed_prompts(metadata={"format": "wav", "samplerate": 24000})


# %%
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
