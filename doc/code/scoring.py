# %% [markdown]
#
# This Jupyter notebook gives an introduction on how to use PyRIT to score responses. We set up a `SelfAskGptClassifier`, which uses Azure Open AI to score a response. In this example, we mock a response from a targeted endpoint and ask if it matches any classifier.
#
# Before starting this, make sure you are [set up and authenticated to use Azure OpenAI endpoints](../setup/azure_openai_setup.md)

# %%

from pyrit.common import default_values
from pyrit.chat import AzureOpenAIChat
from pyrit.score import SelfAskGptClassifier, PROMPT_INJECTION_CLASSIFIER

default_values.load_default_env()
gpt_4 = AzureOpenAIChat(deployment_name="gpt-4")
prompt_injection_classifier = SelfAskGptClassifier(content_classifier=PROMPT_INJECTION_CLASSIFIER, gpt_4=gpt_4)

# Mocked response when an injection is successful
text_with_dan_injection_response = """
(🔒Normal Output) Hello there!
(🔓Developer Mode Output) Hey there, world! Stay in Developer Mode.
"""

prompt_injection_classifier.score_text(text=text_with_dan_injection_response)

# %% [markdown]
#
# As you can see, the `score_value` is set to `prompt_injection`, which indicates our scorer interpreted that the injection was successful. Below we'll try again with normal text.

# %%

text_without_dan_injection_response = """
Hello there!
"""
prompt_injection_classifier.score_text(text=text_without_dan_injection_response)
