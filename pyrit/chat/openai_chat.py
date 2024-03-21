
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pyrit.common import default_values

from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

from pyrit.chat.openai_base import OpenAIBase

class OpenAIChat(OpenAIBase):
    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        model: str = None,
    ) -> None:
        base_url = default_values.get_required_value(
            env_var_name=super().ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=base_url
        )
        api_key = default_values.get_required_value(
            env_var_name=super().API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        _client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        _async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        _model = model

        super().__init__(client=_client, async_client=_async_client, model=_model) 
