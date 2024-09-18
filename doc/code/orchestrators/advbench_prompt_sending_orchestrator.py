# %% [markdown]
# # AdvBench PromptSendingOrchestrator
#
# This demo is uses prompts from the AdvBench dataset to try against a target. It includes the ways you can send the prompts,
# and how you can view results. Before starting, import the necessary libraries.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/).
#
# The first example is as simple as it gets.

# %%
import time
import uuid

from pyrit.prompt_target import AzureOpenAIGPT4OChatTarget
from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.datasets import fetch_adv_bench_dataset

default_values.load_default_env()

target = AzureOpenAIGPT4OChatTarget()

# You could optionally pass memory labels to orchestrators, which will be associated with each prompt and assist in retrieving or scoring later.
test_op_name = str(uuid.uuid4())
test_user_name = str(uuid.uuid4())
memory_labels = {"op_name": test_op_name, "user_name": test_user_name}
with PromptSendingOrchestrator(prompt_target=target, memory_labels=memory_labels) as orchestrator:
    adv_bench_prompts = fetch_adv_bench_dataset()

    start = time.time()
    await orchestrator.send_prompts_async(prompt_list=adv_bench_prompts.prompts[:3])  # type: ignore
    end = time.time()

    print(f"Elapsed time for operation: {end-start}")

    memory = orchestrator.get_memory()

    for entry in memory:
        print(entry)

# %% [markdown]
# ### Introducing Rate Limit (RPM) Threshold
#
# Some targets have a specific Rate Limit (Requests Per Minute) they can handle. In order to abide by this limitation
# and avoid exceptions, you can configure `max_requests_per_minute` on the target before using it with an orchestrator.
#
# **Note**: `batch_size` should be 1 to properly use the RPM provided.

# %%
import time

from pyrit.prompt_target import AzureOpenAIGPT4OChatTarget
from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.datasets import fetch_adv_bench_dataset


default_values.load_default_env()
max_requests_per_minute = 20

target = AzureOpenAIGPT4OChatTarget(max_requests_per_minute=max_requests_per_minute)

with PromptSendingOrchestrator(prompt_target=target, batch_size=1) as orchestrator:
    adv_bench_prompts = fetch_adv_bench_dataset()

    start = time.time()
    await orchestrator.send_prompts_async(prompt_list=adv_bench_prompts.prompts[:3])  # type: ignore
    end = time.time()

    print(f"Elapsed time for operation, with request delay is: {end-start}")
    assert (end - start) > (60 / max_requests_per_minute * len(adv_bench_prompts.prompts[:3]))

# %% [markdown]
# ### Adding Converters
#
# Additionally, we can make it more interesting by initializing the orchestrator with different types of prompt converters.
# This variation takes the original example, but converts the text to base64 before sending it to the target.

# %%
from pyrit.prompt_target import AzureOpenAIGPT4OChatTarget

from pyrit.datasets import fetch_adv_bench_dataset
from pyrit.common import default_values
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter


default_values.load_default_env()

target = AzureOpenAIGPT4OChatTarget()

with PromptSendingOrchestrator(prompt_target=target, prompt_converters=[Base64Converter()]) as orchestrator:

    adv_bench_prompts = fetch_adv_bench_dataset()

    # this is run in a Jupyter notebook, so we can use await
    await orchestrator.send_prompts_async(prompt_list=adv_bench_prompts.prompts[:3])  # type: ignore

    memory = orchestrator.get_memory()

    for entry in memory:
        print(entry)
