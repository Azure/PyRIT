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

# %%
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest, NormalizerRequestPiece

# text_prompt_to_send = "Hi what is 2+2?"

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
orchestrator = PromptSendingOrchestrator(objective_target=target)

await orchestrator.send_normalizer_requests_async(prompt_request_list=[normalizer_request])  # type: ignore

await orchestrator.print_conversations_async()  # type: ignore
