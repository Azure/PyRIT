# %% [markdown]
# # Human in the Loop Scoring with Gradio
# This example shows how to use the Gradio UI to perform human-in-the-loop scoring.

# %%
import uuid

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece
from pyrit.score import HumanInTheLoopScorerGradio

initialize_pyrit(memory_db_type=IN_MEMORY)
memory = CentralMemory.get_memory_instance()
scorer = HumanInTheLoopScorerGradio() # This will start the Gradio UI and can only be created once per notebook.

# %%
# This cell can be run multiple times to simulate multiple scoring requests.

promptRequestPiece = PromptRequestPiece(
    role="assistant",
    original_value="The quick brown fox jumps over the lazy dog.",
    prompt_target_identifier="MockTarget",
    conversation_id="MockConversation"
)

result = await scorer.score_async(promptRequestPiece)
print(result)

# %%
scorer.__del__()
memory.dispose_engine()
