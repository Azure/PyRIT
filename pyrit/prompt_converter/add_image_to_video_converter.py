# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from pathlib import Path

import numpy as np

from pyrit.common.path import DB_DATA_PATH
from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


# Choose the codec based on extension
video_encoding_map = {
    "mp4": "mp4v",
    "avi": "XVID",
    "mov": "MJPG",
    "mkv": "X264",
}


class AddImageVideoConverter(PromptConverter):
    """
    Adds an image to a video at a specified position.
    Also, currently the image is placed in the whole video, not at a specific timepoint

    Args:
        video_path (str): File path of video to add image to
        output_path (str, Optional): File path of output video. Defaults to None.
        img_position (tuple, Optional): Position to place image in video. Defaults to (10, 10).
        img_resize_size (tuple, Optional): Size to resize image to. Defaults to (500, 500).
    """

    def __init__(
        self,
        video_path: str,
        output_path: str = None,
        img_position: tuple = (10, 10),
        img_resize_size: tuple = (500, 500),
    ):

        if not video_path:
            raise ValueError("Please provide valid video path")

        self._output_path = output_path
        self._img_position = img_position
        self._img_resize_size = img_resize_size
        self._video_path = video_path

    async def _add_image_to_video(self, image_path: str, output_path: str) -> str:
        """
        Adds image to video
        Args:
            image_path (str): The image path to add to video.
            output_path (str): The output video path.

        Returns:
            output_path (str): The output video path.
        """

        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as e:
            logger.error("Could not import opencv. You may need to install it via 'pip install pyrit[opencv]'")
            raise e

        if not image_path:
            raise ValueError("Please provide valid image path value")

        input_image_data = data_serializer_factory(
            category="prompt-memory-entries", data_type="image_path", value=image_path
        )
        input_video_data = data_serializer_factory(
            category="prompt-memory-entries", data_type="video_path", value=self._video_path
        )

        # Open the video to ensure it exists
        video_bytes = await input_video_data.read_data()

        azure_storage_flag = input_video_data._is_azure_storage_url(self._video_path)
        video_path = self._video_path

        try:
            if azure_storage_flag:
                # If the video is in Azure storage, download it first

                # Save the video bytes to a temporary file
                local_temp_path = Path(DB_DATA_PATH, "temp_video.mp4")
                with open(local_temp_path, "wb") as f:
                    f.write(video_bytes)
                video_path = str(local_temp_path)

            cap = cv2.VideoCapture(video_path)

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            file_extension = video_path.split(".")[-1].lower()
            if file_extension in video_encoding_map:
                video_char_code = cv2.VideoWriter_fourcc(*video_encoding_map[file_extension])  # type: ignore
                output_video = cv2.VideoWriter(output_path, video_char_code, fps, (width, height))
            else:
                raise ValueError(f"Unsupported video format: {file_extension}")

            # Load and resize the overlay image

            input_image_bytes = await input_image_data.read_data()
            image_np_arr = np.frombuffer(input_image_bytes, np.uint8)
            overlay = cv2.imdecode(image_np_arr, cv2.IMREAD_UNCHANGED)
            overlay = cv2.resize(overlay, self._img_resize_size)

            # Get overlay image dimensions
            image_height, image_width, _ = overlay.shape
            x, y = self._img_position  # Position where the overlay will be placed

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Ensure overlay fits within the frame boundaries
                if x + image_width > width or y + image_height > height:
                    logger.info("Overlay image is too large for the video frame. Resizing to fit.")
                    overlay = cv2.resize(overlay, (width - x, height - y))
                    image_height, image_width, _ = overlay.shape

                # Blend overlay with frame
                if overlay.shape[2] == 4:  # Check number of channels on image
                    alpha_overlay = overlay[:, :, 3] / 255.0
                    for c in range(0, 3):
                        frame[y : y + image_height, x : x + image_width, c] = (
                            alpha_overlay * overlay[:, :, c]
                            + (1 - alpha_overlay) * frame[y : y + image_height, x : x + image_width, c]
                        )
                else:
                    frame[y : y + image_height, x : x + image_width] = overlay

                # Write the modified frame to the output video
                output_video.write(frame)

        finally:
            # Release everything
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()
            if azure_storage_flag:
                os.remove(local_temp_path)

        logger.info(f"Video saved as {output_path}")

        return output_path

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "image_path") -> ConverterResult:
        """
        Converter that adds an image to a video

        Args:
            prompt (str): The image file name to be added to the video.
            input_type (PromptDataType): type of data
        Returns:
            ConverterResult: The filename of the converted video as a ConverterResult Object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        output_video_serializer = data_serializer_factory(category="prompt-memory-entries", data_type="video_path")

        if not self._output_path:
            output_video_serializer.value = await output_video_serializer.get_data_filename()
        else:
            output_video_serializer.value = self._output_path

        # Add video to the image
        updated_video = await self._add_image_to_video(image_path=prompt, output_path=output_video_serializer.value)
        return ConverterResult(output_text=str(updated_video), output_type="video_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "video_path"
