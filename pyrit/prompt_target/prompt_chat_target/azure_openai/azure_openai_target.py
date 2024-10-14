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
from pyrit.prompt_target.openai.openai_common_text_interface import OpenAICommonTextInterface

logger = logging.getLogger(__name__)



class AzureOpenAITarget(OpenAICommonTextInterface):

    deployment_model = Literal["Gpt-3.5", "Gpt-4.0", "Gpt-4.0o", "GPT-4v", "Dall-e", "Completion"]

    def __init__(
        self,
        *,
        deployment_model: deployment_model = "Gpt-4.0o",
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        use_aad_auth: bool = False,
        memory: MemoryInterface = None,
        api_version: str = "2024-02-01",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Class that initializes an Azure OpenAI chat target. This class facilitates text as input and output

        Read more about the various models here:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models.


        Args:
            deployment_model (deployment_model, optional): The model to use for the deployment.
            deployment_name (str, optional): The name of the deployment. Defaults to the
                AZURE_OPENAI_CHAT_DEPLOYMENT environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_ENDPOINT environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_KEY environment variable.
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

        if deployment_model == "Gpt-3.5":
            self.api_key_environment_variable = "AZURE_OPENAI_GPT3_5_KEY"
            self.endpoint_uri_environment_variable = "AZURE_OPENAI_GPT3_5_ENDPOINT"
            self.deployment_environment_variable = "AZURE_OPENAI_GPT3_5_DEPLOYMENT"
        elif deployment_model == "Gpt-4.0":
            self.api_key_environment_variable = "AZURE_OPENAI_GPT4_0_KEY"
            self.endpoint_uri_environment_variable = "AZURE_OPENAI_GPT4_0_ENDPOINT"
            self.deployment_environment_variable = "AZURE_OPENAI_GPT4_0_DEPLOYMENT"
        elif deployment_model == "Gpt-4.0o":
            self.api_key_environment_variable = "AZURE_OPENAI_GPT4_0_O_KEY"
            self.endpoint_uri_environment_variable = "AZURE_OPENAI_GPT4_0_O_ENDPOINT"
            self.deployment_environment_variable = "AZURE_OPENAI_GPT4_0_O_DEPLOYMENT"
        elif deployment_model == "GPT-4v":
            self.api_key_environment_variable = "AZURE_OPENAI_GPT4_V_KEY"
            self.endpoint_uri_environment_variable = "AZURE_OPENAI_GPT4_V_ENDPOINT"
            self.deployment_environment_variable = "AZURE_OPENAI_GPT4_V_DEPLOYMENT"
        elif deployment_model == "Dall-e":
            self.api_key_environment_variable = "AZURE_OPENAI_DALLE_KEY"
            self.endpoint_uri_environment_variable = "AZURE_OPENAI_DALLE_ENDPOINT"
            self.deployment_environment_variable = "AZURE_OPENAI_DALLE_DEPLOYMENT"
        elif deployment_model == "Completion":
            self.api_key_environment_variable = "AZURE_OPENAI_COMPLETION_KEY"
            self.endpoint_uri_environment_variable = "AZURE_OPENAI_COMPLETION_ENDPOINT"
            self.deployment_environment_variable = "AZURE_OPENAI_COMPLETION_DEPLOYMENT"
        else:
            raise ValueError(f"Invalid deployment model: {deployment_model}")

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.deployment_environment_variable, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        )

        if use_aad_auth:
            logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")
            token_provider = get_token_provider_from_default_azure_credential()

            self._async_client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
        else:
            api_key = default_values.get_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=api_key
            )

            self._async_client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )