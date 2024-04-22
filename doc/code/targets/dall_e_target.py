# %% [markdown]
# ## Image Target Demo
# This notebook demonstrates how to use the image target to create an image from a text-based prompt

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid

from pyrit.common import default_values
from pyrit.models import PromptRequestPiece
from pyrit.prompt_target import DALLETarget
from pyrit.prompt_target.dall_e_target import ResponseFormat

test_conversation_id = str(uuid.uuid4())

prompt_to_send = "Draw me a racoon pirate as a French artist in France"

default_values.load_default_env()

img_prompt_target = DALLETarget(
    deployment_name=os.environ.get("AZURE_DALLE_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_DALLE_ENDPOINT"),
    api_key=os.environ.get("AZURE_DALLE_API_KEY"),
    api_version="2024-02-01",
    response_format=ResponseFormat.URL,
)

# %%

request = PromptRequestPiece(
    role="user",
    original_prompt_text=prompt_to_send,
).to_prompt_request_response()

image_resp = img_prompt_target.send_prompt(prompt_request=request).request_pieces[0]

print(f"image location: {image_resp.converted_prompt_text}")
print(f"metadata: {image_resp.prompt_metadata}")

# %% [markdown]
# ### Downloading and viewing the genereated image:
# The `download_image` function will save the image locally and return back the location of the saved image. It is already called from within the `send_prompt` function and stored within the response. The value is shown below:

# %%
from PIL import Image

image_location = image_resp.converted_prompt_text

im = Image.open(image_location)
im.show()

img_prompt_target.dispose_db_engine()

# %%
