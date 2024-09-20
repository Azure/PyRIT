import pathlib
from pyrit.common.path import DATASETS_PATH
from pyrit.models.prompt_template import PromptTemplate
from pyrit.prompt_converter.fuzzer_converter.fuzzer_converter_base import FuzzerConverter
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget


class FuzzerSimilarConverter(FuzzerConverter):
    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: PromptTemplate = None):
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "similar_converter.yaml"
            )
        )
        super().__init__(converter_target=converter_target, prompt_template=prompt_template)