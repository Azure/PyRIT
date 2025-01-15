# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # REALTIME TARGET

# %% [markdown]
# ## Using PyRIT

# %%
from pyrit.prompt_target import RealtimeTarget
from pyrit.common import initialize_pyrit, IN_MEMORY

initialize_pyrit(memory_db_type=IN_MEMORY)

target = RealtimeTarget()

# %% [markdown]
# ## Single Turn Audio Conversation

# %%

from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest, NormalizerRequestPiece

prompt_to_send = "test_rt_audio1.wav"

normalizer_request = NormalizerRequest(
    request_pieces=[
        NormalizerRequestPiece(
            prompt_value=prompt_to_send,
            prompt_data_type="audio_path",
        ),
    ]
)

# %%
await target.connect()  # type: ignore

orchestrator = PromptSendingOrchestrator(objective_target=target)

await orchestrator.send_normalizer_requests_async(prompt_request_list=[normalizer_request])  # type: ignore

await orchestrator.print_conversations_async()  # type: ignore

# %% [markdown]
# ## Single Turn Text Conversation

# %%
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator


await target.connect()  # type: ignore
prompt_to_send = "Give me an image of a raccoon pirate as a Spanish baker in Spain"

request = PromptRequestPiece(
    role="user",
    original_value=prompt_to_send,
).to_prompt_request_response()


orchestrator = PromptSendingOrchestrator(objective_target=target)
response = await orchestrator.send_prompts_async(prompt_list=[prompt_to_send])  # type: ignore
await orchestrator.print_conversations_async()  # type: ignore

# %% [markdown]
# ## Multiturn Text Conversation

# %%

from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest, NormalizerRequestPiece

await target.connect()  # type: ignore

text_prompt_to_send = "Hi what is 2+2?"

normalizer_request = NormalizerRequest(
    request_pieces=[
        NormalizerRequestPiece(
            prompt_value=text_prompt_to_send,
            prompt_data_type="text",
        )
    ]
)

second_prompt_to_send = "Now add 2?"
second_normalizer_request = NormalizerRequest(
    request_pieces=[
        NormalizerRequestPiece(
            prompt_value=second_prompt_to_send,
            prompt_data_type="text",
        )
    ]
)


# %%
orchestrator = PromptSendingOrchestrator(objective_target=target)

await orchestrator.send_normalizer_requests_async(prompt_request_list=[normalizer_request])  # type: ignore

await orchestrator.print_conversations_async()  # type: ignore

# %%

# %%
orchestrator = PromptSendingOrchestrator(objective_target=target)

await orchestrator.send_normalizer_requests_async(prompt_request_list=[second_normalizer_request])  # type: ignore

await orchestrator.print_conversations_async()  # type: ignore

# %%
await target.disconnect()  # type: ignore

# %%
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
