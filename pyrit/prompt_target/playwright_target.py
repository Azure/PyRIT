# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Protocol

from pyrit.models import (
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.prompt_target.common.prompt_target import PromptTarget

# Avoid errors for users who don't have playwright installed
if TYPE_CHECKING:
    from playwright.async_api import Page
else:
    Page = None


class InteractionFunction(Protocol):
    """
    Defines the structure of interaction functions used with PlaywrightTarget.
    """

    async def __call__(self, page: "Page", prompt_request: PromptRequestResponse) -> str: ...


class PlaywrightTarget(PromptTarget):
    """
    PlaywrightTarget uses Playwright to interact with a web UI.

    The interaction function receives the complete PromptRequestResponse and can process
    multiple pieces as needed. All pieces must be of type 'text' or 'image_path'.

    Parameters:
        interaction_func (InteractionFunction): The function that defines how to interact with the page.
            This function receives the Playwright page and the complete PromptRequestResponse.
        page (Page): The Playwright page object to use for interaction.
    """

    # Supported data types
    SUPPORTED_DATA_TYPES = {"text", "image_path"}

    def __init__(
        self,
        *,
        interaction_func: InteractionFunction,
        page: "Page",
    ) -> None:
        super().__init__()
        self._interaction_func = interaction_func
        self._page = page

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        if not self._page:
            raise RuntimeError(
                "Playwright page is not initialized. Please pass a Page object when initializing PlaywrightTarget."
            )

        try:
            text = await self._interaction_func(self._page, prompt_request)
        except Exception as e:
            raise RuntimeError(f"An error occurred during interaction: {str(e)}") from e

        # For response construction, we'll use the first piece as reference
        # but the interaction function can process all pieces
        request_piece = prompt_request.request_pieces[0]
        response_entry = construct_response_from_request(request=request_piece, response_text_pieces=[text])
        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if not prompt_request.request_pieces:
            raise ValueError("This target requires at least one prompt request piece.")

        # Validate that all pieces are supported types
        for i, piece in enumerate(prompt_request.request_pieces):
            piece_type = piece.converted_value_data_type
            if piece_type not in self.SUPPORTED_DATA_TYPES:
                supported_types = ", ".join(self.SUPPORTED_DATA_TYPES)
                raise ValueError(
                    f"This target only supports {supported_types} prompt input. " f"Piece {i} has type: {piece_type}."
                )
