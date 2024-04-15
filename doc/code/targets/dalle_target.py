# %% [markdown]
# ## Image Target Demo
# This notebook demonstrates how to use the image target to create an image from a text-based prompt

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
import json

from pyrit.common import default_values
from pyrit.prompt_target import DallETarget
from pyrit.prompt_target.dall_e_target import ResponseFormat

test_conversation_id = str(uuid.uuid4())

prompt_to_send = "Draw me a racoon pirate as a French artist in France"

# %% [markdown]
# ## Using DALL-E model

# %%
default_values.load_default_env()

img_prompt_target = DallETarget(
    deployment_name=os.environ.get("AZURE_DALLE_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_DALLE_ENDPOINT"),
    api_key=os.environ.get("AZURE_DALLE_API_KEY"),
    api_version="2024-02-01",
    response_format=ResponseFormat.URL,
)

# %%


image_resp = img_prompt_target.send_prompt(
    normalized_prompt=prompt_to_send,
    conversation_id=test_conversation_id,
    normalizer_id=test_normalizer_id,
)

# %% [markdown]
# ### Viewing the response:

# %%
image_resp_dict = json.loads(image_resp)
image_resp_dict

# %% [markdown]
# ### Downloading and viewing the genereated image:
# The `download_image` function will save the image locally and return back the location of the saved image. It is already called from within the `send_prompt` function and stored within the response. The value is shown below:

# %%
image_location = image_resp_dict["image_file_location"]

# %% [markdown]
# The `download_image` function can be called on its own as well using an image url and output filename

# %%
downloaded_image_location = img_prompt_target.download_image(
    image_url=image_resp_dict["data"][0]["url"], output_filename="image0.png"
)

# %% [markdown]
# The image can be viewed using the code snippet below:

# %%
from PIL import Image

im = Image.open(image_location)
im.show()

# %%
img_prompt_target.dispose_db_engine()
