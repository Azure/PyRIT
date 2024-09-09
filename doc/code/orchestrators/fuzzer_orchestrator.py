# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fuzzer orchestrator
#
# This notebooks showcases the fuzzer orchestrator. Starting from an initial set of prompt templates, it attempts to generate new prompt templates by applying modifications. The generated templates are kept around if they produce new jailbreaks on the input set of prompts. This is based on the paper "GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts." (https://arxiv.org/pdf/2309.10253) by Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing.

# %%
from pyrit.models import PromptDataset, PromptTemplate
from pyrit.prompt_converter import ShortenConverter, ExpandConverter
from pyrit.orchestrator import FuzzerOrchestrator
from pyrit.common.path import DATASETS_PATH
from pyrit.prompt_target import AzureOpenAITextChatTarget
from pyrit.prompt_converter import ShortenConverter, ExpandConverter
import pathlib

prompts = PromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompts" / "illegal.prompt")

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
scoring_target=AzureOpenAITextChatTarget()

# %%
converter_target=AzureOpenAITextChatTarget()
prompt_shorten_converter = ShortenConverter(converter_target=converter_target)
prompt_expand_converter = ExpandConverter(converter_target=converter_target)

# %%

fuzzer_orchestrator = FuzzerOrchestrator(
    prompts = prompts.prompts,
    prompt_target = target,
    prompt_templates = prompt_templates,
    scoring_target= scoring_target,
    template_converters = [prompt_shorten_converter, prompt_expand_converter],
    jailbreak_goal=10,
    verbose=True
)

result = await fuzzer_orchestrator.execute_fuzzer() 
print('Success:', result.success)
print('Description: ', result.description)
print('Templates', "\n".join(result.templates))


# %%
from pyrit.memory import DuckDBMemory

memory = DuckDBMemory()

for conversation_id in result.prompt_target_conversation_ids:
    conversation = memory._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)
    print("\n---------------------------\n")
    print(f"Conversation ID: {conversation_id}\n")
    for piece in conversation:
        print(f"{piece.sequence} - {piece.role} - {piece.converted_value}\n\n")

# %%
fuzzer_orchestrator._initial_prompts_nodes[1].children[1].children[0].children[0].children

# %%
