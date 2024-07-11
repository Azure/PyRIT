# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import concurrent.futures
import enum
import json
import logging

from httpx import HTTPStatusError
from openai import BadRequestError

from pyrit.common import default_values, net_utility
from pyrit.exceptions.exception_classes import handle_bad_request_exception
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptTarget


logger = logging.getLogger(__name__)


class CrucibleTarget(PromptTarget):
    API_KEY_ENVIRONMENT_VARIABLE: str = "CRUCIBLE_API_KEY"


    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = None,
        memory: MemoryInterface = None,
    ) -> None:
        self._memory = memory if memory else DuckDBMemory()


        self._endpoint = endpoint
        self._api_key : str = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )


    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        try:
            response = await self._complete_text_async(request.converted_value)
            response_entry = construct_response_from_request(
                request=request,
                response_text_pieces=[response.text]
            )
        except HTTPStatusError as bre:
            if bre.response.status_code == 400:
                response_entry = handle_bad_request_exception(
                    response_text=bre.response.text,
                    request=request,
                    is_content_filter=True
                )


        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    async def _complete_text_async(self, text: str) -> str:
        payload: dict[str, str] = {
            "data": text,
        }

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=f"{self._endpoint.rstrip('/')}/score",
            method="POST",
            request_body=payload,
            headers={"Authorization": self._api_key},
        )

        if not resp.text:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp.text}"')
        return resp
