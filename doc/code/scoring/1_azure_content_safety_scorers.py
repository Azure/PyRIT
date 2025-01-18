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
#
# # 1. Float Scale Scoring using Azure Content Safety API
#
# The Azure Content Safety API is one of our most reliable scorers for detecting harms. Although it isn't very flexible, it's extremely fast and reliable and can be used to score images or text.
#
# In order to use this API, you need to configure a few environment variables:
#
# - AZURE_CONTENT_SAFETY_API_ENDPOINT: The endpoint for the Azure Content Safety API
# - AZURE_CONTENT_SAFETY_API_KEY: The API key for the Azure Content Safety API (if not using AAD Auth)
#
# As an alternative to key-based authentication, you may set `use_aad_auth=True` and use identity-based authentication.
#
# Note that this api returns a value between 0 and 7. This is different from likert scales, which return a value between 1 and 5. Because both are `float_scale` scores, these values are all normalized to floating point values between 0.0 and 1.0 and can be directly compared. This is sometimes interesting as an operator e.g. if there are scenarios where a `SelfAskLikertScorer` and `AzureContentFilterScorer` produce very different values.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).

# %%
import os

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import AzureContentFilterScorer

initialize_pyrit(memory_db_type=IN_MEMORY)


# Set up the Azure Content Filter
azure_content_filter = AzureContentFilterScorer(
    # Comment out either api_key or use_aad_auth
    api_key=os.environ.get("AZURE_CONTENT_SAFETY_API_KEY"),
    # use_aad_auth=True,
    endpoint=os.environ.get("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
)

response = PromptRequestPiece(
    role="assistant",
    original_value_data_type="text",
    original_value="I hate you.",
)
memory = CentralMemory.get_memory_instance()
# need to write it manually to memory as score table has a foreign key constraint
memory.add_request_response_to_memory(request=PromptRequestResponse([response]))

# Run the request
scores = await azure_content_filter.score_async(response)  # type: ignore
assert scores[0].get_value() > 0  # azure_severity should be value 2 based on the documentation

for score in scores:
    # score_metadata contains azure_severity original value
    print(f"{score} {score.score_metadata}")

# %%
memory.dispose_engine()
