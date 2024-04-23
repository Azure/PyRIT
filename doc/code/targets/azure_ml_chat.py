# %% [markdown]
# # Introduction
#
# This code shows how to use Azure Machine Learning (AML) managed online endpoints with PyRIT.
#
# ## Prerequisites
#
# 1. **Deploy an AML-Managed Online Endpoint:** Confirm that an Azure Machine Learning managed online endpoint is
#      already deployed.
#
# 1. **Obtain the API Key:**
#    - Navigate to the AML Studio.
#    - Go to the 'Endpoints' section.
#    - Retrieve the API key and endpoint URI.
#    <br> <img src="./../../assets/aml_managed_online_endpoint_api_key.png" alt="aml_managed_online_endpoint_api_key.png" height="400"/> <br>
#
# 1. **Set the Environment Variable:**
#    - Add the obtained API key to an environment variable named `AZURE_ML_KEY`.
#    - Add the obtained endpoint URI to an environment variable named `AZURE_ML_MANAGED_ENDPOINT`.
#
# ## Create a AzureMLChatTarget
#
# After deploying a model and populating your env file, creating an endpoint is as simple as the following

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common import default_values
from pyrit.models import PromptRequestPiece
from pyrit.prompt_target import AzureMLChatTarget


default_values.load_default_env()

request = PromptRequestPiece(
    role="user",
    original_prompt_text="Hello world!",
).to_prompt_request_response()

with AzureMLChatTarget() as azure_ml_chat_target:
    print(azure_ml_chat_target.send_prompt(prompt_request=request))


# %% [markdown]
#
# You can then use this cell anywhere you would use a `PromptTarget` object.
# For example, you can create a red teaming orchestrator and do the entire [Gandalf Demo](../demo/1_gandalf.ipynb) but use this AML model.
# This is also shown in the [Multiturn Demo](../demo/2_multiturn_strategies.ipynb).
