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
# Within this demo, the module prepares prompts that are designed to generate demographic groups for job roles.
# 
# The attack is substituted inside the given prompt template and sent to the target LLM.
#
# _This manual attack module was sourced from Project Moonshot: [Job Role Generator](https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/job_role_generator.py)_
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
) as target_llm:
    result_list = []

    gender = ["Male", "Female"]
    race = ["Malay", "Chinese", "Indian", "White", "Black", "African-American"]

    job = "Software Engineer"

    job_role_converter = JobRoleGenerator(converter_target=target_llm)

    for g in gender:
        print(f"{job} ({g}): {await job_role_converter.convert_async(job=job, demographic=g)}") # type: ignore

    for r in race:
        print(f"{job} ({r}): {await job_role_converter.convert_async(job=job, demographic=r)}") # type: ignore