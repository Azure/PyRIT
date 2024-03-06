# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: pyrit_kernel
#     language: python
#     name: pyrit_kernel
# ---

# %% [markdown]
# ## Introduction
#
# This Jupyter notebook gives an introduction on how to use PyRIT to abstract dealing
# with different [embedding](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/understand-embeddings) and [completion](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/completions) endpoints, and gives an introduction
# for how to add custom endpoints for your own use cases.
#
# There will be three main parts to this notebook:
# 1. Work with Azure OpenAI Completions
# 2. Work with Azure OpenAI Embeddings
# 3. Embeddings serialization
#
# Before starting this, make sure you are [set up and authenticated to use Azure OpenAI endpoints](../setup/setup_azure.md)

# %%
import os
from pyrit.common import default_values

default_values.load_default_env()

api_base = os.environ.get("AZURE_OPENAI_COMPLETION_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_COMPLETION_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_COMPLETION_DEPLOYMENT")

# %% [markdown]
# ## Azure OpenAI Completions
#
# Once you are configured, then you will be able to get completions for your text. The
# `complete_text()` response returns a wrapper for the OpenAI completion API that will allow you to
# easily get all the different fields.

# %%

from pprint import pprint
from pyrit.completion.azure_completions import AzureCompletion


prompt = "hello world!"

engine = AzureCompletion(api_key=api_key, endpoint=api_base, deployment=deployment_name)

text_response = engine.complete_text(text=prompt)

pprint(text_response, width=280, compact=True)
