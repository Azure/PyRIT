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
#
# This notebooks shows how to interact with the Realtime Target to send text or audio prompts and receive back an audio output and the text transcript of that audio

# %%
from pyrit.prompt_target import RealtimeTarget
from pyrit.common import initialize_pyrit, IN_MEMORY

initialize_pyrit(memory_db_type=IN_MEMORY)

target = RealtimeTarget()

# %% [markdown]
# ## Audio Conversation
#
# The following shows how to interact with the Realtime Target with audio files as your prompt. You can either use pre-made audio files with the pcm16 format or you can use PyRIT converters to help turn your text into audio.

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

await target.disconnect()  # type: ignore

# %% [markdown]
# ## Text Conversation
#
# This section below shows how to interact with the Realtime Target with text prompts

# %%
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator


await target.connect()  # type: ignore
prompt_to_send = "What is the capitol of France?"

request = PromptRequestPiece(
    role="user",
    original_value=prompt_to_send,
).to_prompt_request_response()


orchestrator = PromptSendingOrchestrator(objective_target=target)
response = await orchestrator.send_prompts_async(prompt_list=[prompt_to_send])  # type: ignore
await orchestrator.print_conversations_async()  # type: ignore

# %%
await target.disconnect()  # type: ignore

# %%
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
