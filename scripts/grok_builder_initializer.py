import os

from pyrit.common.apply_defaults import set_default_value, set_global_variable
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer


class GrokBuilderInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Grok Builder Initializer"

    @property
    def description(self) -> str:
        return "Registers a Grok chat target and uses it as the default helper model for prompt building."

    @property
    def required_env_vars(self) -> list[str]:
        return [
            "OPENAI_CHAT_ENDPOINT",
            "OPENAI_CHAT_KEY",
            "OPENAI_CHAT_MODEL",
        ]

    async def initialize_async(self) -> None:
        endpoint = os.environ["OPENAI_CHAT_ENDPOINT"]
        api_key = os.environ["OPENAI_CHAT_KEY"]
        model_name = os.environ["OPENAI_CHAT_MODEL"]
        underlying_model = os.getenv("OPENAI_CHAT_UNDERLYING_MODEL") or None

        target_kwargs = {
            "endpoint": endpoint,
            "api_key": api_key,
            "model_name": model_name,
        }
        if underlying_model:
            target_kwargs["underlying_model"] = underlying_model

        builder_target = OpenAIChatTarget(**target_kwargs)

        set_global_variable(name="default_converter_target", value=builder_target)
        set_default_value(
            class_type=PromptConverter,
            parameter_name="converter_target",
            value=builder_target,
        )
