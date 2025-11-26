# %%NBQA-CELL-SEP52c935
from pyrit.datasets import fetch_llm_latent_adversarial_training_harmful_dataset
from pyrit.executor.attack import (
    AttackExecutor,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.prompt_target import TextTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# Fetch prompt column from harmful datasets
prompt_dataset = fetch_llm_latent_adversarial_training_harmful_dataset()
# Some fetch functions may include parameters such as the below example for unsafe prompts
# prompt_dataset = fetch_pku_safe_rlhf_dataset(False)

# Use the first 8 examples for red teaming
prompt_list = list(prompt_dataset.get_values(first=8))

# Send prompts using the attack and capture responses
attack = PromptSendingAttack(objective_target=prompt_target)

results = await AttackExecutor().execute_multi_objective_attack_async(  # type: ignore
    attack=attack,
    objectives=prompt_list,
)

for result in results:
    await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %%NBQA-CELL-SEP52c935
from pyrit.datasets import fetch_decoding_trust_stereotypes_dataset
from pyrit.prompt_target import TextTarget

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
examples_source = (
    "https://raw.githubusercontent.com/AI-secure/DecodingTrust/main/data/stereotype/dataset/user_prompts.csv"
)

attack = PromptSendingAttack(objective_target=prompt_target)

# Fetch examples from DecodingTrust 'Stereotype' dataset using the 'targeted' system prompt and topics of "driving" and "technology"
prompt_dataset = fetch_decoding_trust_stereotypes_dataset(
    examples_source,
    source_type="public_url",
    stereotype_topics=["driving", "technology"],
    target_groups=None,
    system_prompt_type="targeted",
)

# Use the first 4 examples
prompt_list = list(prompt_dataset.get_values(first=4))

results = await AttackExecutor().execute_multi_objective_attack_async(  # type: ignore
    attack=attack,
    objectives=prompt_list,
)

for result in results:
    await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %%NBQA-CELL-SEP52c935
from pyrit.datasets.harmbench_dataset import fetch_harmbench_dataset

dataset = fetch_harmbench_dataset()

# %%NBQA-CELL-SEP52c935
dataset.harm_categories
