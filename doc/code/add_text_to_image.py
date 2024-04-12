# In[1]:
from pyrit.prompt_converter import AddTextImageConverter
from pyrit.common.path import RESULTS_PATH
from PIL import Image
import pathlib


# In[2]:
image_converter = AddTextImageConverter(
    input_file=str(pathlib.Path(RESULTS_PATH / "images" / "roakey.png")),
    output_file=str(pathlib.Path(RESULTS_PATH / "images" / "roakey_text.png")),
)


# In[3]:
output_image_file = image_converter.convert(prompt="Hello, I am a friendly and helpful raccoon named Roakey!")

output_image = Image.open(output_image_file)
output_image.show()


# In[ ]:
