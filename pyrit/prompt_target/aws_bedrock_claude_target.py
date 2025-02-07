import logging
import json
import boto3
from typing import Optional
import asyncio

from botocore.exceptions import ClientError

from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

class AWSBedrockClaudeTarget(PromptTarget):
    """
    This class initializes an AWS Bedrock target for any of the Anthropic Claude models.
    Local AWS credentials (typically stored in ~/.aws) are used for authentication.
    See the following for more information: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    Parameters:
        model_id (str): The model ID for target claude model
        max_tokens (int): maximum number of tokens to generate
        temperature (float, optional): The amount of randomness injected into the response.
        top_p (float, optional): Use nucleus sampling
        top_k (int, optional): Only sample from the top K options for each subsequent token
        verify (bool, optional): whether or not to perform SSL certificate verification
    """

    def __init__(
        self,
        *,
        model_id: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        verify: bool = True,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        super().__init__(max_requests_per_minute=max_requests_per_minute)

        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._verify = verify

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        response = await self._complete_text_async(request.converted_value)

        response_entry = construct_response_from_request(request=request, response_text_pieces=[response])

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    async def _complete_text_async(self, text: str) -> str:
        brt = boto3.client(service_name="bedrock-runtime", verify=self._verify)

        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ]
        }

        if self._temperature:
            native_request['temperature'] = self._temperature
        if self._top_p:
            native_request['top_p'] = self._top_p
        if self._top_k:
            native_request['top_k'] = self._top_k

        request = json.dumps(native_request)

        try:
            #response = brt.invoke_model(modelId=self._model_id, body=request)
            response = await asyncio.to_thread(brt.invoke_model, modelId=self._model_id, body=request)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self._model_id}'. Reason: {e}")
            exit()

        model_response = json.loads(response["body"].read())

        answer = model_response["content"][0]["text"]

        logger.info(f'Received the following response from the prompt target "{answer}"')
        return answer
