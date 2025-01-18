# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Protocol

from pyrit.models import (
    PromptRequestPiece,
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

    async def __call__(self, page: "Page", request_piece: PromptRequestPiece) -> str: ...


class PlaywrightTarget(PromptTarget):
    """
    PlaywrightTarget uses Playwright to interact with a web UI.

    Parameters:
        interaction_func (InteractionFunction): The function that defines how to interact with the page.
        page (Page): The Playwright page object to use for interaction.
    """

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

        request_piece = prompt_request.request_pieces[0]

        try:
            text = await self._interaction_func(self._page, request_piece)
        except Exception as e:
            raise RuntimeError(f"An error occurred during interaction: {str(e)}") from e

        response_entry = construct_response_from_request(request=request_piece, response_text_pieces=[text])
        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
