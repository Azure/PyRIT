# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import AzureOpenAI

from pyrit.common import default_values
from pyrit.embedding._text_embedding import _TextEmbedding


class AzureTextEmbedding(_TextEmbedding):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"

    def __init__(
        self, *, api_key: str = None, endpoint: str = None, deployment: str = None, api_version: str = "2024-02-01"
    ) -> None:
        """Generate embedding using the Azure API

        Args:
            api_key: The API key to use. Defaults to environment variable.
            endpoint: The API base to use, sometimes referred to as the api_base. Defaults to environment variable.
            deployment: The engine to use, in AOAI referred to as deployment, in some APIs referred to as model.
                        Defaults to environment variable.
            api_version: The API version to use. Defaults to "2024-02-01".
        """

        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        deployment = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment
        )

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
        )
        self._model = deployment
        super().__init__()
