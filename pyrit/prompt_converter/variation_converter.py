from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter
from pyrit.models import PromptTemplate, ChatMessage
import pathlib
from pyrit.common.path import DATASETS_PATH

class VariationConverter(PromptConverter):
    def __init__(self, converterLLM: PromptTarget = None, promptTemplate: PromptTemplate = None):
        self.converterLLM = converterLLM #TODO: make a default, if not given
        self.promptTemplate = promptTemplate
        # set to default strategy if not provided
        self.promptTemplate = (
            promptTemplate
            if promptTemplate
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH)
                / "attack_strategies"
                / "prompt_variation"
                / "prompt_variation.yaml"
            )
        )
        

    def convert(self, prompts: list[str]) -> list[str]:
        """
        Creates variations of a prompt by sending to LLM
        """
        return_messages = []
        for prompt in prompts:
            system_prompt = self.promptTemplate.apply_custom_metaprompt_parameters(prompt=prompt)

            chat_entries = [ChatMessage(role="system", content=system_prompt)]
            chat_entries.append(ChatMessage(role="user", content=""))

            response_msg = self.converterLLM.complete_chat(messages=chat_entries)
            return_messages.append(response_msg)
            
        return return_messages