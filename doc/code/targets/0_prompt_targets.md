# Prompt Targets

Prompt Targets are endpoints for where to send prompts. For example, a target could be a GPT-4 or Llama endpoint. Targets are typically used with other components like [orchestrators](../orchestrators/0_orchestrator.md), [scorers](../scoring/0_scoring.md), and [converters](../converters/0_converters.ipynb).

- An orchestrator's main job is to change prompts to a given format, apply any converters, and then send them off to prompt targets (sometimes using various strategies). Within an orchestrator, prompt targets are (mostly) swappable, meaning you can use the same logic with different target endpoints.
- A scorer's main job is to score a prompt. Often, these use LLMs, in which case, a given scorer can often use different configured targets.
- A converter's job is to transform a prompt. Often, these use LLMs, in which case, a given converter can use different configured targets.

Prompt targets are found [here](https://github.com/Azure/PyRIT/tree/main/pyrit/prompt_target/) in code.

The main entry method follow the following signature:

```
async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
```

A `PromptRequestResponse` object is a normalized object with all the information a target will need to send a prompt. This is discussed in more depth [here](../memory/3_prompt_request.md).
