# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## AML Chat Targets
#
# This code shows how to use Azure Machine Learning (AML) managed online endpoints with PyRIT.
#
# ### Prerequisites
#
# 1. **Deploy an AML-Managed Online Endpoint:** Confirm that an Azure Machine Learning managed online endpoint is
#      already deployed.
#
# 1. **Obtain the API Key:**
#    - Navigate to the AML Studio.
#    - Go to the 'Endpoints' section.
#    - Retrieve the API key and endpoint URI.
#    <br> <img src="../../../assets/aml_managed_online_endpoint_api_key.png" alt="aml_managed_online_endpoint_api_key.png" height="400"/> <br>
#
# 1. **Set the Environment Variable:**
#    - Add the obtained API key to an environment variable named `AZURE_ML_KEY`.
#    - Add the obtained endpoint URI to an environment variable named `AZURE_ML_MANAGED_ENDPOINT`.
#
# ### Create a AzureMLChatTarget
#
# After deploying a model and populating your env file, creating an endpoint is as simple as the following

# %%
from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureMLChatTarget

default_values.load_default_env()


azure_ml_chat_target = AzureMLChatTarget()

with PromptSendingOrchestrator(prompt_target=azure_ml_chat_target) as orchestrator:
    response = await orchestrator.send_prompts_async(prompt_list=["Hello world!"])  # type: ignore
    print(response[0])

# %% [markdown]
#
# You can then use this cell anywhere you would use a `PromptTarget` object.
# For example, you can create a red teaming orchestrator and use this instead of the `AzureOpenAI` target and do the [Gandalf or Crucible Demos](./3_custom_targets.ipynb) but use this AML model.
#
# This is also shown in the [Red Teaming Orchestrator](../orchestrators/3_red_teaming_orchestrator.ipynb) documentation.
