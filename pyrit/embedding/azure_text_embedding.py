# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import AzureOpenAI

from pyrit.common import environment_variables
from pyrit.embedding._text_embedding import _TextEmbedding


class AzureTextEmbedding(_TextEmbedding):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"

    def __init__(
        self, *, api_key: str = None, endpoint: str = None, deployment: str = None, api_version: str = "2023-05-15"
    ) -> None:
        """Generate embedding using the Azure API

        Args:
            api_key: The API key to use
            endpoint: The API base to use
            deployment: The engine to use, usually name of the deployment
            api_version: The API version to use
        """

        api_key = environment_variables.get_required_value(self.API_KEY_ENVIRONMENT_VARIABLE, api_key)
        endpoint = environment_variables.get_required_value(self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, endpoint)
        deployment = environment_variables.get_required_value(self.DEPLOYMENT_ENVIRONMENT_VARIABLE, deployment)

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
        )
        self._model = deployment
        super().__init__()
