# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import Literal, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI, BadRequestError
from openai.types.chat import ChatCompletion

from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
from pyrit.common import default_values
from pyrit.exceptions import EmptyResponseException, PyritException
from pyrit.exceptions import pyrit_target_retry, handle_bad_request_exception
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OpenAITarget(PromptChatTarget):

    ADDITIONAL_REQUEST_HEADERS: str = "OPENAI_ADDITIONAL_REQUEST_HEADERS"


    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        headers: str = None,
        is_azure_target = True,
        use_aad_auth: bool = False,
        memory: MemoryInterface = None,
        api_version: str = "2024-06-01",
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Abstract class that initializes an Azure OpenAI chat target.

        Read more about the various models here:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models.


        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                AZURE_OPENAI_CHAT_DEPLOYMENT environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_ENDPOINT environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_KEY environment variable.
            headers (str, optional): Headers of the endpoint (JSON).
            use_aad_auth (bool, optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
            api_version (str, optional): The version of the Azure OpenAI API. Defaults to
                "2024-02-01".
            max_tokens (int, optional): The maximum number of tokens to generate in the response.
                Defaults to 1024.
            temperature (float, optional): The temperature parameter for controlling the
                randomness of the response. Defaults to 1.0.
            top_p (float, optional): The top-p parameter for controlling the diversity of the
                response. Defaults to 1.0.
            frequency_penalty (float, optional): The frequency penalty parameter for penalizing
                frequently generated tokens. Defaults to 0.5.
            presence_penalty (float, optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history. Defaults to 0.5.
            max_requests_per_minute (int, optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
        """
        PromptChatTarget.__init__(self, memory=memory, max_requests_per_minute=max_requests_per_minute)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._extra_headers: dict = {}
        try:
            request_headers = default_values.get_required_value(
                env_var_name=self.ADDITIONAL_REQUEST_HEADERS, passed_value=headers
            )
            if isinstance(request_headers, str):
                try:
                    self._extra_headers = json.loads(request_headers)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
        except ValueError:
            logger.info("No headers have been passed, setting empty default headers")



        self._api_version = api_version
        self._is_azure_target = is_azure_target

        if self._is_azure_target:
             self._initialize_azure_vars(deployment_name, endpoint, api_key, use_aad_auth)
        else:
            self._initialize_openai_vars(deployment_name, endpoint, api_key)


    def _initialize_azure_vars(
              self,
              deployment_name: str,
              endpoint: str,
              api_key: str,
              use_aad_auth: bool
        ):
        self._set_azure_openai_env_configuration_vars()

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.deployment_environment_variable, passed_value=deployment_name
        )
        self._endpoint = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        ).rstrip("/")

        if use_aad_auth:
                logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")
                token_provider = get_token_provider_from_default_azure_credential()

                self._async_client = AsyncAzureOpenAI(
                    azure_ad_token_provider=token_provider,
                    api_version=self._api_version,
                    azure_endpoint=self._endpoint,
                    default_headers=self._extra_headers,
                )
        else:
            self._api_key = default_values.get_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=api_key
            )

            self._async_client = AsyncAzureOpenAI(
                api_key=self._api_key,
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                default_headers=self._extra_headers,
            )

    def _initialize_openai_vars(
              self,
              deployment_name: str,
              endpoint: str,
              api_key: str
    ):
        self._api_key = default_values.get_required_value(
            env_var_name="OPENAI_CHAT_KEY", passed_value=api_key
        )

        # Any available model. See https://platform.openai.com/docs/models
        self._deployment_name = default_values.get_required_value(
            env_var_name="OPENAI_CHAT_DEPLOYMENT", passed_value=deployment_name
        )

        endpoint = endpoint if endpoint else "https://api.openai.com/v1/chat/completions"

        self._async_client = AsyncOpenAI(
            api_key=self._api_key,
            default_headers=self._extra_headers,
        )




    @abstractmethod
    def _set_azure_openai_env_configuration_vars(self) -> None:
        """
        Sets deployment_environment_variable, endpoint_uri_environment_variable, and api_key_environment_variable
        which are read from .env
        """
        pass