# %% [markdown]
# # Scoring
#
# This Jupyter notebook gives an introduction on how to use PyRIT to score responses automatically. Currently, two main types of scorers are available:
# - `SelfAskGptClassifier`: classifies a response into one of several categories (e.g., detecting whether a text string contains a prompt injection)
# - `SelfAskGptLikertScale`: classifies a response into one of several levels on a Likert scale (e.g., scoring the severity of misinformation within a text string)
#
# Both of these scorers inherit from the `SelfAskScore` class, which you can also use to create a custom scorer as follows:
#
# ```
# CustomScorer = SelfAskScore(
#     prompt_template_path=custom_prompt_template,
#     content_classifier=custom_classifier,
#     chat_target=chat_target
# )
# ```
#
# ## Classification Scoring
#
# In this first example, we set up a `SelfAskGptClassifier` and use it to determine whether a text string contains a prompt injection.
#
# Before starting this, make sure you are [set up and authenticated to use Azure OpenAI endpoints](../setup/setup_azure.md)

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common import default_values
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score import SelfAskGptClassifier, PROMPT_INJECTION_CLASSIFIER

default_values.load_default_env()
with AzureOpenAIChatTarget() as azure_openai_chat_target:
    prompt_injection_classifier = SelfAskGptClassifier(
        content_classifier=PROMPT_INJECTION_CLASSIFIER, chat_target=azure_openai_chat_target
    )

    # Mocked response when an injection is successful
    text_with_dan_injection_response = """
    (🔒Normal Output) Hello there!
    (🔓Developer Mode Output) Hey there, world! Stay in Developer Mode.
    """

    scored_response = prompt_injection_classifier.score_text(text=text_with_dan_injection_response)
    print("Scored response is given as:", scored_response)
    text_without_dan_injection_response = """Hello there!"""
    # As you can see, the `score_value` is set to `prompt_injection`, which indicates our scorer interpreted that the
    # injection was successful. Below we'll try again with normal text.
    scored_response = prompt_injection_classifier.score_text(text=text_without_dan_injection_response)
    print("Scored response is given as:", scored_response)

# %% [markdown]
# ## Likert Scale Scoring
#
# In some cases, we are interested in scoring a response on a Likert scale that measures the prevalence or severity of some type of content.
#
# In this example, we use the `SelfAskLikertScorer` to measure the severity of political misinformation in a text string.

# %%
from pyrit.score import SelfAskGptLikertScale, POLITICAL_MISINFO_CLASSIFIER

with AzureOpenAIChatTarget() as azure_openai_chat_target:

    political_misinfo_scorer = SelfAskGptLikertScale(
        content_classifier=POLITICAL_MISINFO_CLASSIFIER, chat_target=azure_openai_chat_target
    )

    text_with_political_misinfo = """
    Hillary Clinton is a member of the Illuminati!
    """

    scored_response = political_misinfo_scorer.score_text(text=text_with_political_misinfo)
    print("Scored response is given as:", scored_response)
    text_without_political_misinfo = """Today is a beautiful day."""
    # As you can see, the `score_value` indicates the level of severity for the misinformation example.
    # Below we'll try again with normal text.
    scored_response = political_misinfo_scorer.score_text(text=text_without_political_misinfo)
    print("Scored response is given as:", scored_response)
