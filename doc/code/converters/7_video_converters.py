# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
# # Adding Images to a Video
#
# Adds an image to a video.
# To use this converter you'll need to install opencv which can be done with
# `pip install pyrit[opencv]`

# %%
import pathlib

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_converter import AddImageVideoConverter

initialize_pyrit(memory_db_type=IN_MEMORY)

input_video = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "sample_video.mp4")
input_image = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "pyrit_architecture.png")

video = AddImageVideoConverter(video_path=input_video)
converted_vid = await video.convert_async(prompt=input_image, input_type="image_path")  # type: ignore
converted_vid
