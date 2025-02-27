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

# %%
import pathlib

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_converter import AddImageVideoConverter

initialize_pyrit(memory_db_type=IN_MEMORY)

input_video = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "sample_video.mp4")
output_video = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "sample_output_video.mp4")
input_image = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "ice_cream.png")

video = AddImageVideoConverter(video_path=input_video, img_resize_size=(100, 100))
converted_vid = await video.convert_async(prompt=input_image, input_type="image_path")
converted_vid
