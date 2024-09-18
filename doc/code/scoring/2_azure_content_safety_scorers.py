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
#     name: pyrit-311
# ---

# %% [markdown]
#
# ## Float Scale Scoring using Azure Content Safety API
#
# The Azure Content Safety API is one of our most reliable scorers for detecting harms. Although it isn't very flexible, it's extremely fast and reliable and can be used to score images or text.
#
# In order to use this API, you need to configure a few environment variables:
#
# - AZURE_CONTENT_SAFETY_API_ENDPOINT: The endpoint for the Azure Content Safety API
# - AZURE_CONTENT_SAFETY_API_KEY: The API key for the Azure Content Safety API
#
# Note that this api returns a value between 0 and 7. This is different from likert scales, which return a value between 1 and 5. Because both are `float_scale` scores, these values are all normalized to floating point values between 0.0 and 1.0 and can be directly compared. This is sometimes interesting as an operator e.g. if there are scenarios where a `SelfAskLikertScorer` and `AzureContentFilterScorer` produce very different values.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/).

# %%
import os
from pyrit.score import AzureContentFilterScorer
from pyrit.common import default_values
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.memory import DuckDBMemory


default_values.load_default_env()
memory = DuckDBMemory()

# Set up the Azure Content Filter
azure_content_filter = AzureContentFilterScorer(
    api_key=os.environ.get("AZURE_CONTENT_SAFETY_API_KEY"),
    endpoint=os.environ.get("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
    memory=memory,
)

response = PromptRequestPiece(
    role="assistant",
    original_value_data_type="text",
    original_value="I hate you.",
)

# need to write it manually to memory as score table has a foreign key constraint
memory.add_request_response_to_memory(request=PromptRequestResponse([response]))

scores = await azure_content_filter.score_async(response)  # type: ignore
assert azure_content_filter.get_azure_severity(scores[0].get_value()) > 1  # should be value 2 base on the documentation

for score in scores:
    print(score)
