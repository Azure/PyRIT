# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from openai import AzureOpenAI

from pyrit.auth.azure_auth import AzureAuth, get_default_scope
from pyrit.common import default_values
from pyrit.embedding._text_embedding import _TextEmbedding


class AzureTextEmbedding(_TextEmbedding):
    """
    Provide text embedding functionality using Azure OpenAI services.
    """

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2024-02-01",
        use_entra_auth: bool = False,
    ) -> None:
        """
        Generate embedding using the Azure API. Authenticate with either an API key or Entra authentication.

        Args:
            api_key: The API key to use (only if you're not using Entra authentication). Defaults to
                environment variable.
            endpoint: The API base to use, sometimes referred to as the api_base. Defaults to environment variable.
            deployment: The engine to use, in AOAI referred to as deployment, in some APIs referred to as model.
                        Defaults to environment variable.
            api_version: The API version to use. Defaults to "2024-02-01".
            use_entra_auth: Whether to use Entra authentication. Defaults to False.

        Raises:
            ValueError: If using Entra ID auth and an api_key is also provided.
        """
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        deployment = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment
        )
        if use_entra_auth:
            if api_key:
                raise ValueError("If using Entra ID auth, please do not specify api_key.")
            scope = get_default_scope(endpoint)
            token = AzureAuth(token_scope=scope).get_token()
            self._client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                azure_ad_token=token,
            )
        else:
            api_key = default_values.get_required_value(
                env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
            )
            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=deployment,
            )

        self._model = deployment
        super().__init__()
