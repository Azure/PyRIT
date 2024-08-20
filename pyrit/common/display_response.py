# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from PIL import Image
from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.models import PromptRequestPiece


logger = logging.getLogger(__name__)


def display_response(response_piece: PromptRequestPiece) -> None:
    """Displays response images if running in notebook environment.

    Args:
        response_piece (PromptRequestPiece): The response piece to display.
    """
    if (
        response_piece.response_error == "none"
        and response_piece.converted_value_data_type == "image_path"
        and is_in_ipython_session()
    ):
        with open(response_piece.converted_value, "rb") as f:
            img = Image.open(f)
            # Jupyter built-in display function only works in notebooks.
            display(img)  # type: ignore # noqa: F821
    if response_piece.response_error == "blocked":
        logger.info("---\nContent blocked, cannot show a response.\n---")
