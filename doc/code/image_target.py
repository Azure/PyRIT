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

# %% [markdown]
# ## Image Target Demo
# This notebook demonstrates how to use the image target to create an image from a text-based prompt

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
PROMPT_TO_SEND = "Draw me a cute baby red panda cuddling a cute pillow pet"

# %%
import os
import uuid

from pyrit.common import default_values
from pyrit.prompt_target import ImageTarget

# When using a Prompt Target with an Orchestrator, conversation ID and normalizer ID are handled for you
test_conversation_id = str(uuid.uuid4())
test_normalizer_id = "1"

# %% [markdown]
# ## Using DALLE model

# %%
default_values.load_default_env()

img_prompt_target = ImageTarget(
    deployment_name="pyrit_dall-e-3",
    endpoint=os.environ.get("AZURE_DALLE_ENDPOINT"),
    api_key=os.environ.get("AZURE_DALLE_API_KEY"),
    api_version="2024-02-01",
    response_format="url",
)

# %%
image_resp = img_prompt_target.send_prompt(
    normalized_prompt=PROMPT_TO_SEND,
    conversation_id=test_conversation_id,
    normalizer_id=test_normalizer_id,
)

# %% [markdown]
# ### Viewing the response:

# %%
image_resp

# %% [markdown]
# ### Downloading and viewing the genereated image:
# The `download_image` function will save the image locally and return back the location of the saved image. It is already called from within the `send_prompt` function and stored within the response. The value is shown below:

# %%
image_location = image_resp["image_file_location"]

# %% [markdown]
# The `download_image` function can be called on its own as well using an image url and output filename

# %%
downloaded_image_location = img_prompt_target.download_image(
    image_url=image_resp["data"][0]["url"], output_filename="image0.png"
)

# %% [markdown]
# The image can be viewed using the code snippet below:

# %%
from PIL import Image

im = Image.open(image_location)
im.show()
