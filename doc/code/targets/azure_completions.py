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
# Before starting this, make sure you are [set up and authenticated to use Azure OpenAI endpoints](../../setup/populating_secrets.md)

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from pyrit.prompt_target import AzureOpenAICompletionTarget
from pyrit.models import PromptRequestPiece


request = PromptRequestPiece(
    role="user",
    original_value="Hello! Who are you?",
).to_prompt_request_response()

with AzureOpenAICompletionTarget(api_key=api_key, endpoint=api_base, deployment_name=deployment_name) as target:
    response = await target.send_prompt_async(prompt_request=request)  # type: ignore
    pprint(response.request_pieces[0].converted_value, width=280, compact=True)

# %%
