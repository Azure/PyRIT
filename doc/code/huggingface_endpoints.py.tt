# %% [markdown]
# # Introduction

# This code shows how to use Hugging Face managed online endpoints with PyRIT. Hugging Face support is currently experimental
# and may not work as intended.

# ## Prerequisites

# Before beginning, ensure that you have the model id obtained from the Hugging Face as shown below.
# <br> <img src="./../../assets/huggingface_model_id.png" alt="huggingface_model_id.png" height="400"/> <br>

# After deploying a model and populating your env file, creating an endpoint is as simple as the following
# %%

from pyrit.common import default_values

from pyrit.models import ChatMessage
from pyrit.chat import HuggingFaceChat


default_values.load_default_env()

model_id = "Fredithefish/Guanaco-3B-Uncensored-v2"
huggingface_chat_engine = HuggingFaceChat(model_id=model_id)

# %% [markdown]
#
# After the model is created, you can use it like any other `ChatSupport` object. For example, you can complete a chat as shown below.

# %%
huggingface_chat_engine.complete_chat(messages=[ChatMessage(role="user", content="Hello world!")])
