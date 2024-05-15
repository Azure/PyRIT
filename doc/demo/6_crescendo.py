# %% [markdown]
# ## Crescendo Attacks
# Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications.
# These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms.
# However, a recent line of attacks, known as "jailbreaks", seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow
# the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo.
# Unlike existing jailbreak methods, Crescendo is a multi-turn jailbreak that interacts with the model in a seemingly benign manner.
# It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's
# replies, progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro,
# Gemini-Ultra, LlaMA-2 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack
# success rates across all evaluated models and tasks. Furthermore, we introduce Crescendomation, a tool that automates the Crescendo attack,
# and our evaluation showcases its effectiveness against state-of-the-art models.
# https://arxiv.org/abs/2404.01833
#
# This notebook illustrates the usage of the Crescendo attack in PyRIT.
# 1. We define an objective
# 2. We pick a model to be attacked (in this case an Azure OpenAI Chat model
# 3. We execute the Crescendo attack and obtain the final result
# 4. We output the entire conversation from memory
#
