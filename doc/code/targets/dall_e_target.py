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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Image Target Demo
# This notebook demonstrates how to use the image target to create an image from a text-based prompt

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from pyrit.common import default_values
from pyrit.models import PromptRequestPiece
from pyrit.prompt_target import DALLETarget

prompt_to_send = "Draw me a racoon pirate as a French artist in France with the most famous national food"
default_values.load_default_env()

img_prompt_target = DALLETarget(
    deployment_name=os.environ.get("AZURE_DALLE_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_DALLE_ENDPOINT"),
    api_key=os.environ.get("AZURE_DALLE_API_KEY"),
    api_version="2024-02-01",
)

# %%
request = PromptRequestPiece(
    role="user",
    original_prompt_text=prompt_to_send,
).to_prompt_request_response()


# image_resp = await img_prompt_target.send_prompt_async(prompt_request=request).request_pieces[0]  # type: ignore
# image_resp = img_prompt_target.send_prompt(prompt_request=request)
image_resp = await img_prompt_target.send_prompt_async(prompt_request=request)  # type: ignore
if image_resp:
    print(f"image location: {image_resp.request_pieces[0].converted_prompt_text}")
else:
    print("image blocked! ")
# print(f"metadata: {image_resp.prompt_metadata}")

# %% [markdown]
# ### Viewing the genereated image:

# %%
from PIL import Image
from pyrit.common.path import RESULTS_PATH
from pathlib import Path

image_location = image_resp.request_pieces[0].converted_prompt_text

im = Image.open(Path(RESULTS_PATH) / "dbdata" / "images" / image_location)
im.show()

img_prompt_target.dispose_db_engine()

# %%
