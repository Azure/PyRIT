# %%NBQA-CELL-SEP52c935
from pyrit.executor.attack import (
    AttackConverterConfig,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.prompt_converter import StringJoinConverter, VariationConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import OpenAIChatTarget, TextTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

objective = "tell me how to cut down a tree"

# converter target to send prompt to; in this case configured to the Azure OpenAI GPT-4o Chat model
converter_target = OpenAIChatTarget()
prompt_variation_converter = VariationConverter(converter_target=converter_target)

converters = PromptConverterConfiguration.from_converters(
    converters=[prompt_variation_converter, StringJoinConverter()]
)

converter_config = AttackConverterConfig(request_converters=converters)

target = TextTarget()
attack = PromptSendingAttack(
    objective_target=target,
    attack_converter_config=converter_config,
)

result = await attack.execute_async(objective=objective)  # type: ignore

printer = ConsoleAttackResultPrinter()
await printer.print_conversation_async(result=result)  # type: ignore
