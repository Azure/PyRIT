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
#    - Retrieve the API key and endpoint URI from the 'Consume' tab
#    <br> <img src="../../../assets/aml_managed_online_endpoint_api_key.png" alt="aml_managed_online_endpoint_api_key.png" height="400"/> <br>
#
# 1. **Set the Environment Variable:**
#    - Go to 'Model ID' section under the 'Details' tab and click the link to find the model name at the top of the page (e.g., mistralai-Mistral-7B-Instruct-v01)
#    - Add the obtained API key to an environment variable named `{MODEL_NAME}_KEY` with all letters capitalized and dashes replaced with underscores
#      (e.g., `MISTRALAI_MISTRAL_7B_INSTRUCT_V01_KEY` or `PHI_3_MINI_4K_INSTRUCT_KEY`).
#    - Add the obtained endpoint URI to an environment variable named `{MODEL_NAME}_ENDPOINT` with all letters capitalized and dashes replaced with underscores
#      (e.g., `MISTRALAI_MISTRAL_7B_INSTRUCT_V01_ENDPOINT` or `PHI_3_MINI_4K_INSTRUCT_ENDPOINT`).
#
# ### Create a AzureMLChatTarget
#
# After deploying a model and populating your env file, send prompts to the model using the `AzureMLChatTarget` class. Model parameters can be passed upon instantiation
# or set using the _set_model_parameters() function. `**param_kwargs` allows for the setting of other parameters not explicitly shown in the constructor. A general list of
# possible adjustable parameters can be found here: https://huggingface.co/docs/api-inference/tasks/text-generation but note that not all parameters may have an effect
# depending on the specific model. The parameters that can be set per model can usually be found in the 'Consume' tab when you click on your endpoint in AML Studio.

# %%
from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureMLChatTarget

default_values.load_default_env()

# Defaults to "mistralai-Mixtral-8x7B-Instruct-v01"
azure_ml_chat_target = AzureMLChatTarget()
# Parameters such as temperature and repetition_penalty can be set using the _set_model_parameters() function.
azure_ml_chat_target._set_model_parameters(temperature=0.9, repetition_penalty=1.3)

with PromptSendingOrchestrator(prompt_target=azure_ml_chat_target) as orchestrator:
    response = await orchestrator.send_prompts_async(prompt_list=["Hello! Describe yourself and the company who developed you."])  # type: ignore
    print(response[0])

# %% [markdown]
#
# You can then use this cell anywhere you would use a `PromptTarget` object.
# For example, you can create a red teaming orchestrator and use this instead of the `AzureOpenAI` target and do the [Gandalf or Crucible Demos](./3_custom_targets.ipynb) but use this AML model.
#
# This is also shown in the [Red Teaming Orchestrator](../orchestrators/3_red_teaming_orchestrator.ipynb) documentation.
