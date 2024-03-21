# ---
# jupyter:
#   jupytext:
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

# %%
import os
import uuid

from pyrit.common import default_values
from pyrit.prompt_target import ImageTarget

# When using a Prompt Target with an Orchestrator, conversation ID and normalizer ID are handled for you
test_conversation_id = str(uuid.uuid4())
test_normalizer_id = "1"

# %%
default_values.load_default_env()

img_prompt_target = ImageTarget(deployment_name = "pyrit_dall-e-3", 
                                endpoint = os.environ.get("AZURE_DALLE_ENDPOINT"), 
                                api_key = os.environ.get("AZURE_DALLE_API_KEY"),  
                                api_version = "2024-02-01")

# %%
image_resp = img_prompt_target.send_prompt(
    prompt="Draw me a baby racoon. It should be the cutest racoon ever. Draw it cuddling a stuffed animal racoon")

# %%
image_resp

# %%
