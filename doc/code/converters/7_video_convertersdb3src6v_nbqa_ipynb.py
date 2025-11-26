# %%NBQA-CELL-SEP52c935
import pathlib

from pyrit.prompt_converter import AddImageVideoConverter
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

input_video = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "sample_video.mp4")
input_image = str(pathlib.Path(".") / ".." / ".." / ".." / "assets" / "pyrit_architecture.png")

video = AddImageVideoConverter(video_path=input_video)
converted_vid = await video.convert_async(prompt=input_image, input_type="image_path")  # type: ignore
converted_vid
