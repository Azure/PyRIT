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
#     display_name: pyrit_kernel
#     language: python
#     name: pyrit_kernel
# ---

# %%
from pyrit.prompt_converter import AddTextImageConverter
from pyrit.common.path import RESULTS_PATH
from PIL import Image
import pathlib


# %%
image_converter = AddTextImageConverter()


# %%
output_image_file = image_converter.convert(
    prompt=str(pathlib.Path(RESULTS_PATH / "images" / "roakey.png")),
    text_to_add="Hello, I am a friendly and helpful raccoon named Roakey!",
)

print(output_image_file)
output_image = Image.open(output_image_file.output_text)
output_image.show()


# %%
