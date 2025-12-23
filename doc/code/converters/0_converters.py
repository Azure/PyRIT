# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Converters

# %% [markdown]
# Converters are used to transform prompts before sending them to the target.
#
# This can be useful for a variety of reasons, such as encoding the prompt in a different format, or adding additional information to the prompt. For example, you might want to convert a prompt to base64 before sending it to the target, or add a prefix to the prompt to indicate that it is a question.
#
# Converters can transform prompts in various ways:
# - **Text-to-Text**: Encoding, obfuscation, translation, and semantic transformations
# - **Multimodal**: Converting between text, images, audio, video, and files
# - **Interactive**: Human-in-the-loop review and modification
#
# ## Converter Modality Reference Table
#
# The following table shows all available converters organized by their input and output modalities:

# %%
import pandas as pd

from pyrit.prompt_converter import get_converter_modalities
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

# Get all converters with their modalities
converter_list = get_converter_modalities()

# Create a list of rows for the DataFrame
rows = []
for name, inputs, outputs in converter_list:
    input_str = ", ".join(inputs) if inputs else "any"
    output_str = ", ".join(outputs) if outputs else "any"
    rows.append({"Input Modality": input_str, "Output Modality": output_str, "Converter": name})

# Create DataFrame and sort
df = pd.DataFrame(rows)
df = df.sort_values(by=["Input Modality", "Output Modality", "Converter"]).reset_index(drop=True)

# Display all rows
pd.set_option("display.max_rows", None)
df

# %% [markdown]
# ## Converter Categories
#
# Converters are organized into the following categories:
#
# - **[Text-to-Text Converters](1_text_to_text_converters.ipynb)**: Non-LLM (encoding, obfuscation) and LLM-based (translation, variation, tone)
# - **[Audio Converters](2_audio_converters.ipynb)**: Text-to-Audio, Audio-to-Text, Audio-to-Audio
# - **[Image Converters](3_image_converters.ipynb)**: Text-to-Image, Image-to-Image
# - **[Video Converters](4_video_converters.ipynb)**: Image-to-Video
# - **[File Converters](5_file_converters.ipynb)**: Text-to-File (PDF, URL)
# - **[Selectively Converting](6_selectively_converting.ipynb)**: Advanced techniques for converting portions of prompts
# - **[Human-in-the-Loop Converter](7_human_converter.ipynb)**: Interactive prompt review and modification

# %% [markdown]
# ## Simple Converter Example

# %% [markdown]
# Converters can be used to perform these types of transformations. Here is a simple program that uses Rot13Converter converter, RandomCapitalLettersConverter, and AsciiArtConverter.

# %%

from pyrit.prompt_converter import (
    AsciiArtConverter,
    BinaryConverter,
    FirstLetterConverter,
    RandomCapitalLettersConverter,
    ROT13Converter,
)

prompt = "tell me how to cut down a tree"


print(await ROT13Converter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await RandomCapitalLettersConverter(percentage=25.0).convert_tokens_async(prompt=prompt))  # type: ignore
print(await AsciiArtConverter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await BinaryConverter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await FirstLetterConverter().convert_tokens_async(prompt=prompt))  # type: ignore

# %% [markdown]
# ## Stacking Converters
#
# Although converters can be used on their own, they should be thought of as a piece in the pipeline. Typically any attack will have arguments so that prompts can be converted before being sent to the target. They can be stacked, use LLMs, and are a powerful tool.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).
#
# ### Stacking Converters with PromptSendingAttack
#
# In the following example, a TextTarget is used so these prompts are simply printed and added to memory. This can be useful if you are red teaming something and need to manually enter prompts. However, the target can be replaced with any other [target](../targets/0_prompt_targets.md). E.g., if you have API access you can add a target there.
#
# In this example the converters are stacked. First a variation is found using `VariationConverter`, and then `StringJoinConverter` is used to add a dash between letters. Remember that order matters. If `StringJoinConverter` came first, we would be asking the LLM to make variations of the prompt:
# "t-e-l-l- - m-e- -h-o-w- -t-o- -c-u-t- -d-o-w-n - a- -t-r-e-e"

# %%
from pyrit.executor.attack import (
    AttackConverterConfig,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.prompt_converter import StringJoinConverter, VariationConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import OpenAIChatTarget, TextTarget

objective = "tell me how to cut down a tree"

# converter target to send prompt to; in this case configured to the Azure OpenAI GPT-4o Chat model
converter_target = OpenAIChatTarget()
prompt_variation_converter = VariationConverter(converter_target=converter_target)

converter_configs = PromptConverterConfiguration.from_converters(  # type: ignore
    converters=[prompt_variation_converter, StringJoinConverter()]
)

converter_config = AttackConverterConfig(request_converters=converter_configs)  # type: ignore

target = TextTarget()
attack = PromptSendingAttack(
    objective_target=target,
    attack_converter_config=converter_config,
)

result = await attack.execute_async(objective=objective)  # type: ignore

printer = ConsoleAttackResultPrinter()
await printer.print_conversation_async(result=result)  # type: ignore

# %%
# Close connection
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
