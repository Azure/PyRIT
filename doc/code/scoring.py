# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scoring
#
# Scoring is a main component of the PyRIT architecture. It is primarily used to evaluate what happens to a prompt. It can be used to help answer questions like:
#
# - Was prompt injection detected?
# - Was the prompt blocked? Why?
# - Was there any harmful content in the response? What was it? How bad was it?
#
# This notebook shows how to use scorers directly.
#
# ### Setup
#
# Before starting this, make sure you are [set up and authenticated to use Azure OpenAI endpoints](../setup/setup_azure.md)
#
# ### True False Scoring
#
# In the simplest case a scorer can answer a question. The following example uses a `SelfAskTrueFalseScorer` to see if prompt injection was successful. This type of scorer is really useful in orchestrators that have to make decisions based on responses.

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common import default_values
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestionPaths


default_values.load_default_env()
with AzureOpenAIChatTarget() as azure_openai_chat_target:
    true_false_scorer = SelfAskTrueFalseScorer(
        true_false_question_path=TrueFalseQuestionPaths.PROMPT_INJECTION.value, chat_target=azure_openai_chat_target
    )

    # Mocked response when an injection is successful
    text_with_dan_injection_response = """
    (🔒Normal Output) Hello there!
    (🔓Developer Mode Output) Hey there, world! Stay in Developer Mode.
    """

    scored_response = (await true_false_scorer.score_text_async(text=text_with_dan_injection_response))[0]  # type: ignore
    print("[Prompt Injection] Scored response is given as:", scored_response, scored_response.score_rationale)

    # Regular Text
    text_without_dan_injection_response = "Hello there!"
    scored_response = (await true_false_scorer.score_text_async(text=text_without_dan_injection_response))[0]  # type: ignore
    print("[Regular Text] Scored response is given as:", scored_response, scored_response.score_rationale)

# %% [markdown]
# ### Classification Scoring
#
# In this example, we set up a `SelfAskCategoryScorer` and use it to determine whether there is harmful content in a response.

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common import default_values
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score import SelfAskCategoryScorer
from pyrit.score.self_ask_category_scorer import ContentClassifierPaths


default_values.load_default_env()
with AzureOpenAIChatTarget() as azure_openai_chat_target:
    harmful_content_classifier = SelfAskCategoryScorer(
        content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value, chat_target=azure_openai_chat_target
    )

    # Mocked response when an injection is successful
    text_with_harmful_content = """
    To cut down a stop sign, you need to use a saw or a chainsaw. It's illegal to do so, but you can do it at night.
    """

    scored_response = (await harmful_content_classifier.score_text_async(text=text_with_harmful_content))[0]  # type: ignore
    print("[Harmful] Scored response is given as:", scored_response.score_category, scored_response.score_rationale)

    # Regular Text
    text_with_no_harmful_content = """
    I'm sorry but I'm afraid I can't answer that. Cutting down a stop sign is illegal.
    """

    scored_response = (await harmful_content_classifier.score_text_async(text=text_with_no_harmful_content))[0]  # type: ignore
    print(
        "[Regular Text] Scored response is given as:", scored_response.score_category, scored_response.score_rationale
    )

# %% [markdown]
# ## Likert Scale Scoring
#
# In some cases, we are interested in scoring a response on a Likert scale that measures the prevalence or severity of some type of content.
#
# In this example, we use the `SelfAskLikertScorer` to measure the severity of political misinformation in a text string.
# Looking at the scored responses, we see that the `text_with_political_misinfo` is scored as 'Severe misinformation', while `text_without_political_misinfo` is scored as 'No misinformation'.

# %%
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score import SelfAskLikertScorer, LikertScalePaths

with AzureOpenAIChatTarget() as self_ask_target:

    political_misinfo_scorer = SelfAskLikertScorer(
        likert_scale_path=LikertScalePaths.MISINFORMATION_SCALE.value, chat_target=self_ask_target
    )

    # Misinformation
    text_with_political_misinfo = "Hillary Clinton is a member of the Illuminati!"
    scored_response = (await political_misinfo_scorer.score_text_async(text=text_with_political_misinfo))[0]  # type: ignore
    print(f"[Misinformation] Scored response is given as: {scored_response} {scored_response.score_rationale}")

    # Regular Text
    text_without_political_misinfo = "Today is a beautiful day."
    scored_response = (await political_misinfo_scorer.score_text_async(text=text_without_political_misinfo))[0]  # type: ignore
    print(f"[Regular Text] Scored response is given as: {scored_response} {scored_response.score_rationale}")

# %% [markdown]
#
# Score text using Azure Content Safety API
#
# In order to use this API, you need to configure a few environment variables:
# AZURE_CONTENT_SAFETY_API_ENDPOINT: The endpoint for the Azure Content Safety API
# AZURE_CONTENT_SAFETY_API_KEY: The API key for the Azure Content Safety API
#


# %% [markdown]
#
# Score text using Azure Content Safety API
#
# In order to use this API, you need to configure a few environment variables:
# AZURE_CONTENT_SAFETY_API_ENDPOINT: The endpoint for the Azure Content Safety API
# AZURE_CONTENT_SAFETY_API_KEY: The API key for the Azure Content Safety API
#


# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from azure.ai.contentsafety.models import TextCategory
from pyrit.score import AzureContentFilter
from pyrit.common import default_values

default_values.load_default_env()


# Set up the Azure Content Filter
azure_content_filter = AzureContentFilter(
    azure_content_safety_key=os.environ.get("AZURE_CONTENT_SAFETY_API_KEY"),
    azure_content_safety_endpoint=os.environ.get("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
    harm_category=TextCategory.HATE,
)

score = azure_content_filter.score_text("I hate you.")
print(score)  # should be value 2 base on the documentation
