# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import cv2
import os

from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class AddImageVideoConverter(PromptConverter):
    """
    Adds an image to a video at a specified position.
    
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
        img_resize_size: tuple =(500,500),   
    ):
        if not video_path:
            raise ValueError("Please provide valid image path")
        
        self.output_path = output_path
        self.img_position = img_position
        self.img_resize_size = img_resize_size
        self._video_path = video_path
        

    def _add_image_to_video(self, image_path: str, output_path: str):
        """
        Adds image to video
        Args:
            Image Path (str): The image path to add to video.

        Returns:
            Image.Image: The image with added text.
        """
        if not image_path:
            raise ValueError("Please provide valid image path value")
        
        if not os.path.exists(image_path):
            print(image_path)
            raise ValueError("Image path does not exist")
        
        if not os.path.exists(self._video_path):
            print(self._video_path)
            raise ValueError("Video path does not exist")

        # Open the video
        cap = cv2.VideoCapture(self._video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Load and resize the overlay image
        overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, self.img_resize_size)

        # Get overlay image dimensions
        h, w, _ = overlay.shape
        x, y = self.img_position  # Position where the overlay will be placed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure overlay fits within the frame boundaries
            if x + w > width or y + h > height:
                print("Overlay image is too large for the video frame. Resizing to fit.")
                overlay = cv2.resize(overlay, (width - x, height - y))

            # Blend overlay with frame
            frame[y:y+h, x:x+w] = overlay

            # Write the modified frame to the output video
            out.write(frame)

        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video saved as {output_path}") 

        return output_path
       

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "image_path") -> ConverterResult:
        """
        Converter that overlays input text on the img_to_add.

        Args:
            prompt (str): The image file name to be added to the video.
            input_type (PromptDataType): type of data
        Returns:
            ConverterResult: The filename of the converted video as a ConverterResult Object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        output_video_serializer = data_serializer_factory(
            category="prompt-memory-entries", data_type="video_path"
        )

        if not self.output_path:
            output_video_serializer.value = output_video_serializer.get_data_filename()
        else:
            output_video_serializer.value = self.output_path
        
        # # Add video to the image
        updated_video = self._add_image_to_video(image_path=prompt, output_path = output_video_serializer.value)
        return ConverterResult(output_text=str(updated_video), output_type="video_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"
