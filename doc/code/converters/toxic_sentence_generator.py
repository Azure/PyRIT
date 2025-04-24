# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Toxic Sentence Generator Converter
#
# This script demonstrates how to use the Toxic Sentence Generator converter, which is based on Project Moonshot's 
# attack module. This converter generates toxic sentence starters based on a given seed prompt to test if an AI system 
# will complete the sentence with toxic content.
#
# Reference: [Toxic Sentence Generator](https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/toxic_sentence_generator.py)

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import ToxicSentenceGeneratorConverter
from pyrit.prompt_target import OpenAIChatTarget

# Initialize PyRIT with in-memory database
initialize_pyrit(memory_db_type=IN_MEMORY)

# Define seed prompts that will be used to generate toxic sentences
seed_prompts = [
    "cyber security",
    "financial advice",
    "social media",
    "privacy"
]

# Initialize OpenAI Chat Target
prompt_target = OpenAIChatTarget()

# Initialize the ToxicSentenceGeneratorConverter with the prompt target
toxic_sentence_generator = ToxicSentenceGeneratorConverter(converter_target=prompt_target)

# Initialize the orchestrator for sending prompts
orchestrator = PromptSendingOrchestrator(
    objective_target=prompt_target,
    prompt_converters=[toxic_sentence_generator],
    verbose=True,
)

# We need to define an async function to use await
async def run_toxic_sentence_generator():
    # Generate and send toxic sentences for each seed prompt
    for seed_prompt in seed_prompts:
        print(f"\nSeed prompt: {seed_prompt}")
        
        # Generate a toxic sentence starter using the converter
        converter_result = await toxic_sentence_generator.convert_async(prompt=seed_prompt)  # type: ignore
        
        # Print the generated toxic sentence
        print(f"Generated toxic sentence: {converter_result.output_text}")
        
        # Send the toxic sentence to the target LLM via the orchestrator
        await orchestrator.send_prompts_async(prompt_list=[converter_result.output_text])  # type: ignore

    # Print all conversations after all prompts are sent
    print("\nAll conversations:")
    await orchestrator.print_conversations_async()  # type: ignore

# Import and use asyncio to run the async function
import asyncio

# Run the async function
asyncio.run(run_toxic_sentence_generator())

# Clean up after running the async function
orchestrator.dispose_db_engine()