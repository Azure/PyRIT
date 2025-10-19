# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Insecure Code Scorer - optional
# This script demonstrates how to use InsecureCodeScorer to evaluate a code snippet for potential security vulnerabilities.
# InsecureCodeScorer uses a language model (LLM) to analyze the code and identify security risks, returning a score based on a predefined threshold.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.models import MessagePiece
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import InsecureCodeScorer

initialize_pyrit(memory_db_type=IN_MEMORY)

# Initialize the LLM model target
chat_target = OpenAIChatTarget()

# Instantiate the scorer with a specified threshold
scorer = InsecureCodeScorer(chat_target=chat_target)

# Example prompt representing a possibly insecure code snippet
example_prompt = """
def authenticate_user(username, password):
    sql = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    execute_sql(sql)
"""

# Create a MessagePiece to hold the example prompt as if it were an assistant response
request = MessagePiece(role="assistant", original_value=example_prompt).to_message()

# Request piece is added to memory first
scorer._memory.add_message_to_memory(request=request)

# Run the scorer to evaluate the security of the prompt
scores = await scorer.score_async(request)  # type: ignore

for score in scores:
    print(f"Score Value: {score.score_value}")
    print(f"Score Rationale: {score.score_rationale}")
    print(f"Score Metadata: {score.score_metadata}")
