# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3. AML Chat Targets
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
#    - Retrieve the API key and endpoint URI from the 'Consume' tab
#    <br> <img src="../../../assets/aml_managed_online_endpoint_api_key.png" alt="aml_managed_online_endpoint_api_key.png" height="400"/> <br>
#
# 1. **Set the Environment Variable:**
#    - Add the obtained API key to an environment variable named `AZURE_ML_KEY`. This is the default API key when the target is instantiated.
#    - Add the obtained endpoint URI to an environment variable named `AZURE_ML_MANAGED_ENDPOINT`. This is the default endpoint URI when the target is instantiated.
#    - If you'd like, feel free to make additional API key and endpoint URI environment variables in your .env file for different deployed models (e.g. mistralai-Mixtral-8x7B-Instruct-v01,
#      Phi-3.5-MoE-instruct, Llama-3.2-3B-Instruct, etc.)
#      and pass them in as arguments to the `_set_env_configuration_vars` function to interact with those models.
#
#
# ## Create a AzureMLChatTarget
#
# After deploying a model and populating your env file, send prompts to the model using the `AzureMLChatTarget` class. Model parameters can be passed upon instantiation
# or set using the _set_model_parameters() function. `**param_kwargs` allows for the setting of other parameters not explicitly shown in the constructor. A general list of
# possible adjustable parameters can be found here: https://huggingface.co/docs/api-inference/tasks/text-generation but note that not all parameters may have an effect
# depending on the specific model. The parameters that can be set per model can usually be found in the 'Consume' tab when you navigate to your endpoint in AML Studio.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureMLChatTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

# Defaults to endpoint and api_key pulled from the AZURE_ML_MANAGED_ENDPOINT and AZURE_ML_KEY environment variables
azure_ml_chat_target = AzureMLChatTarget()

# The environment variable args can be adjusted below as needed for your specific model.
azure_ml_chat_target._set_env_configuration_vars(
    endpoint_uri_environment_variable="AZURE_ML_MANAGED_ENDPOINT", api_key_environment_variable="AZURE_ML_KEY"
)
# Parameters such as temperature and repetition_penalty can be set using the _set_model_parameters() function.
azure_ml_chat_target._set_model_parameters(temperature=0.9, repetition_penalty=1.3)

orchestrator = PromptSendingOrchestrator(objective_target=azure_ml_chat_target)

response = await orchestrator.run_attack_async(objective="Hello! Describe yourself and the company who developed you.")  # type: ignore
await response.print_conversation_async()  # type: ignore

azure_ml_chat_target.dispose_db_engine()

# %% [markdown]
#
# You can then use this cell anywhere you would use a `PromptTarget` object.
# For example, you can create a red teaming orchestrator and use this instead of the `AzureOpenAI` target and do the [Gandalf or Crucible Demos](./2_custom_targets.ipynb) but use this AML model.
#
# This is also shown in the [Red Teaming Orchestrator](../orchestrators/2_multi_turn_orchestrators.ipynb) documentation.
