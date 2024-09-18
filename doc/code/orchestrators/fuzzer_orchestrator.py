# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pyrit_kernel
#     language: python
#     name: pyrit_kernel
# ---

# %%
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataset, PromptTemplate
from pyrit.orchestrator import FuzzerOrchestrator
from pyrit.prompt_converter import ShortenConverter, ExpandConverter
from pyrit.prompt_target import AzureOpenAITextChatTarget

prompts = PromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompts" / "gandalf.prompt")

prompt_template1 = PromptTemplate.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
)
prompt_template2 = PromptTemplate.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aim.yaml"
)
prompt_template3 = PromptTemplate.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aligned.yaml"
)

prompt_templates = [
        prompt_template1.template,
        prompt_template2.template,
        prompt_template3.template
]

# %%
target=AzureOpenAITextChatTarget()

# %%
converter_target=AzureOpenAITextChatTarget()
prompt_shorten_converter = ShortenConverter(converter_target=converter_target)
prompt_expand_converter = ExpandConverter(converter_target=converter_target)

# %%
scoring_target=AzureOpenAITextChatTarget()

# %%

fuzzer_orchestrator = FuzzerOrchestrator(
    prompts = prompts.prompts,
    prompt_target = target,
    prompt_templates = prompt_templates,
    scoring_target= scoring_target,
    template_converter = [prompt_shorten_converter, prompt_expand_converter],
)

result = await fuzzer_orchestrator.execute_fuzzer() 
fuzzer_orchestrator.print_conversation(result=result)
