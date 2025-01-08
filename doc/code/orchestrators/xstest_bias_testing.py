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
# # XSTest Bias Testing - optional
#
# This notebook demonstrates the process of using XSTest Bias Testing examples to evaluate the safety and bias of an AI language model.
# It includes fetching bias testing prompts from the XSTest repository, sending them to the model, and scoring the model's responses based on predefined fairness and bias scales.
# The goal is to assess how well the AI model handles safety without showing exaggerated refusals for safe prompts.


# %%
import os

from pyrit.common import initialize_pyrit, IN_MEMORY
from pyrit.datasets import fetch_xstest_examples
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskLikertScorer, LikertScalePaths


initialize_pyrit(memory_db_type=IN_MEMORY)

prompt_target = OpenAIChatTarget(
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
)
examples_source = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/a3bb396/xstest_v2_prompts.csv"

# Set up the Bias Scorer with the same prompt target to ensure consistency
bias_scorer = SelfAskLikertScorer(
    likert_scale_path=LikertScalePaths.FAIRNESS_BIAS_SCALE.value, chat_target=prompt_target
)


# Fetch examples from XSTest Bias Testing dataset
prompt_dataset = fetch_xstest_examples(examples_source, source_type="public_url")

# Use the first 4 examples
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:4]]

orchestrator = PromptSendingOrchestrator(objective_target=prompt_target, scorers=[bias_scorer])
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
await orchestrator.print_conversations_async()  # type: ignore


# %%
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
