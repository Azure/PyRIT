# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Tuple

import numpy
from PIL import Image

from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class HiddenLayerConverter(PromptConverter):
    """
    Creates a transparency attack by optimizing an alpha channel to blend attack and benign images.

    This converter takes two inputs:
        - Benign image (foreground/target): The harmless image specified during initialization.
        - Attack image (background/harmful): The potentially harmful image passed via the prompt parameter.

    The algorithm optimizes a transparency pattern so that the output PNG exhibits dual perception:
        - On white/light backgrounds: appears as the benign image.
        - On dark backgrounds: reveals the attack image content.
        - AI systems may perceive either image depending on their background processing assumptions.

    Currently, only JPEG images are supported as input. Output images will always be saved as PNG with transparency.

    Note:
        This converter implements the transparency attack as described in:
        `"Transparency Attacks: How Imperceptible Image Layers Can Fool AI Perception"` by
        McKee, F. and Noever, D., 2024: https://arxiv.org/abs/2401.15817

        As stated in the paper: `"The major limitation of the transparency attack is the low success rate when the
        human viewer’s background theme is not light by default or at least a close match to the transparent
        foreground and hidden background layers. When mismatched, the background becomes visible to the human eye
        and the vision algorithm."`
    """

    class AdamOptimizer:
        """
        Implementation of the Adam Optimizer using NumPy.
        The code for this class is taken from the following source:
        https://github.com/xbeat/Machine-Learning/blob/main/Adam%20Optimizer%20in%20Python.md
        """

        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = None
            self.v = None
            self.t = 0

        def update(self, params, grads):
            if self.m is None:
                self.m = numpy.zeros_like(params)
                self.v = numpy.zeros_like(params)

            self.t += 1

            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)  # type: ignore

            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            params -= self.learning_rate * m_hat / (numpy.sqrt(v_hat) + self.epsilon)

            return params

    @staticmethod
    def _validate_input_image(path: str) -> bool:
        """Validates input image to ensure it is a valid JPEG file."""
        return isinstance(path, str) and path.lower().endswith((".jpg", ".jpeg"))

    def __init__(
        self,
        *,
        benign_image_path: Path,
        size: Tuple[int, int] = (150, 150),
        steps: int = 1000,
        learning_rate: float = 0.001,
    ):
        """
        Initializes the converter with the path to a benign image and parameters for blending.

        Args:
            benign_image_path (str): Path to the benign image file. Must be a JPEG file (.jpg or .jpeg).
            size (tuple): Size that the images will be resized to (width, height).
                It is recommended to use a size that matches aspect ratio of both attack and benign images.
                Since the original study resizes images to 150x150 pixels, this is the default size used.
                Bigger values may significantly increase computation time.
            steps (int): Number of optimization steps to perform.
                Recommended range: 100-2000 steps. Default is 1000. Generally, the higher the steps, the
                better end result you can achieve, but at the cost of increased computation time.
            learning_rate (float): Controls the magnitude of adjustments in each step (used by the Adam optimizer).
                Recommended range: 0.0001-0.01. Default is 0.001. Values close to 1 may lead to instability and
                lower quality blending, while values too low may require more steps to achieve a good blend.

        Raises:
            ValueError: If the benign image is invalid or is not in JPEG format.
            ValueError: If the learning rate is not a positive float.
            ValueError: If the size is not a tuple of two positive integers (width, height).
            ValueError: If the steps is not a positive integer.
        """
        self.benign_image_path = benign_image_path
        self.learning_rate = learning_rate
        self.size = size
        self.steps = steps

        if not self._validate_input_image(str(benign_image_path)):
            raise ValueError("Invalid benign image path provided. Only JPEG files are supported as input.")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be a positive float.")
        if not isinstance(size, tuple) or len(size) != 2 or any(dim <= 0 for dim in size):
            raise ValueError(f"Size must be a tuple of two positive integers (width, height). Received {size}")
        if steps <= 0:
            raise ValueError("Steps must be a positive integer.")

    def _load_and_preprocess_image(self, path: str) -> numpy.ndarray:
        """Loads image, converts to grayscale, resizes, and normalizes for optimization."""
        try:
            with Image.open(path) as img:
                img_gray = img.convert("L")  # read as grayscale
                img_resized = img_gray.resize(self.size, Image.Resampling.LANCZOS)
                img_rgb = img_resized.convert("RGB")

                return numpy.array(img_rgb, dtype=numpy.float32) / 255.0  # normalize to [0, 1]
        except Exception as e:
            raise ValueError(f"Failed to load and preprocess image from {path}: {e}")

    def _compute_mse_loss(self, blended_image: numpy.ndarray, target_tensor: numpy.ndarray) -> numpy.floating[Any]:
        """Computes Mean Squared Error (MSE) loss between blended and target images."""
        diff = blended_image - target_tensor
        return numpy.mean(diff**2)

    def _compute_gradients_alpha_layer(
        self,
        blended_image: numpy.ndarray,
        foreground_image: numpy.ndarray,
        background_image: numpy.ndarray,
        white_background: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes gradients to optimize alpha for making the blend resemble the benign image."""
        grad_loss_blended = 2 * (blended_image - foreground_image) / numpy.prod(blended_image.shape)
        grad_blended_alpha = background_image - white_background
        return grad_loss_blended * grad_blended_alpha

    def _create_blended_image(self, attack_image: numpy.ndarray, alpha: numpy.ndarray) -> numpy.ndarray:
        """Creates a blended image using the attack image and alpha transparency."""
        attack_image_uint8 = (attack_image * 255).astype(numpy.uint8)
        transparency_uint8 = (alpha * 255).astype(numpy.uint8)

        # Create RGBA image: 'RGB' from attack image, 'A' creates transparency pattern
        height, width = attack_image_uint8.shape[:2]
        rgba_image = numpy.zeros((height, width, 4), dtype=numpy.uint8)
        rgba_image[:, :, :3] = attack_image_uint8
        rgba_image[:, :, 3] = transparency_uint8[:, :, 0]

        return rgba_image

    async def _save_blended_image(self, attack_image: numpy.ndarray, alpha: numpy.ndarray) -> str:
        """Saves the blended image with transparency as a PNG file."""
        img_serializer = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
        img_serializer.file_extension = "png"

        rgba_image = self._create_blended_image(attack_image, alpha)
        rgba_pil = Image.fromarray(rgba_image, mode="RGBA")
        image_buffer = BytesIO()
        rgba_pil.save(image_buffer, format="PNG")
        image_str = base64.b64encode(image_buffer.getvalue())

        await img_serializer.save_b64_image(data=image_str.decode())
        return img_serializer.value

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "image_path") -> ConverterResult:
        """
        Converts the given prompt by blending an attack image (potentially harmful) with a benign image.
        Uses the Novel Image Blending Algorithm from: https://arxiv.org/abs/2401.15817

        Args:
            prompt (str): The image file path to the attack image.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing path to the manipulated image with transparency.

        Raises:
            ValueError: If the input type is not supported or if the prompt is invalid.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if not self._validate_input_image(prompt):
            raise ValueError("Invalid attack image path provided. Only JPEG files are supported as input.")

        background_image = self._load_and_preprocess_image(prompt)
        foreground_image = self._load_and_preprocess_image(str(self.benign_image_path))

        # Scale attack image by 0.5 to darken it for better blending optimization
        background_tensor = background_image * 0.5

        alpha = numpy.ones_like(background_tensor)  # optimized to determine transparency pattern
        white_background = numpy.ones_like(background_tensor)  # white canvas for blending simulation

        optimizer = self.AdamOptimizer(learning_rate=self.learning_rate)

        for step in range(self.steps):
            # Simulate blending: alpha=1 uses darkened attack image, alpha=0 uses white
            blended_image = alpha * background_tensor + (1 - alpha) * white_background

            loss = self._compute_mse_loss(blended_image, foreground_image)
            if step % 100 == 0:
                logger.debug(f"Step {step}/{self.steps}, Loss: {loss:.4f}")

            # Update alpha to minimize difference between blended and benign image
            grad_alpha = self._compute_gradients_alpha_layer(
                blended_image, foreground_image, background_tensor, white_background
            )
            alpha = optimizer.update(alpha, grad_alpha)
            alpha = numpy.clip(alpha, 0.0, 1.0)

        image_path = await self._save_blended_image(background_tensor, alpha)
        return ConverterResult(output_text=image_path, output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "image_path"
