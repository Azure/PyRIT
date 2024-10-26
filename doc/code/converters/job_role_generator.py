# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---
# %% [markdown]
# # Job Role Generator
#
# The Job Role Generator attack module adds demographic groups to the job role. 
#
# This attack module was sourced from Project Moonshot.
# Reference: [Job Role Generator](https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/job_role_generator.py)
# %%
import os

from pyrit.prompt_converter import JobRoleGenerator
from pyrit.prompt_target import OpenAIChatTarget

from pyrit.common import default_values

default_values.load_default_env()


with OpenAIChatTarget(
    api_key=os.environ.get("OPENAI_KEY"),
    endpoint=os.environ.get("OPENAI_ENDPOINT"),
    deployment_name=os.environ.get("OPENAI_DEPLOYMENT"),
    is_azure_target=False,  # Set to True if using Azure OpenAI
) as attack_llm:

    prompt = "Software Engineer"

    job_role_converter = JobRoleGenerator(converter_target=attack_llm)

    prompts = job_role_converter.manual_attack_demographics(prompt)

    for p in prompts:
        print(f"{p}: {await job_role_converter.convert_async(prompt=p)}") # type: ignore