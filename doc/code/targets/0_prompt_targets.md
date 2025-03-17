# Prompt Targets

Prompt Targets are endpoints for where to send prompts. For example, a target could be a GPT-4 or Llama endpoint. Targets are typically used with other components like [orchestrators](../orchestrators/0_orchestrator.md), [scorers](../scoring/0_scoring.md), and [converters](../converters/0_converters.ipynb).

- An orchestrator's main job is to change prompts to a given format, apply any converters, and then send them off to prompt targets (sometimes using various strategies). Within an orchestrator, prompt targets are (mostly) swappable, meaning you can use the same logic with different target endpoints.
- A scorer's main job is to score a prompt. Often, these use LLMs, in which case, a given scorer can often use different configured targets.
- A converter's job is to transform a prompt. Often, these use LLMs, in which case, a given converter can use different configured targets.

Prompt targets are found [here](https://github.com/Azure/PyRIT/tree/main/pyrit/prompt_target/) in code.


## Send_Prompt_Async

The main entry method follow the following signature:

```
async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
```

A `PromptRequestResponse` object is a normalized object with all the information a target will need to send a prompt, including a way to get a history for that prompt (in the cases that also needs to be sent). This is discussed in more depth [here](../memory/3_prompt_request.md).

## PromptChatTargets vs PromptTargets

A `PromptTarget` is a generic place to send a prompt. With PyRIT, the idea is that it will eventually be consumed by an AI application, but that doesn't have to be immediate. For example, you could have a SharePoint target. Everything you send a prompt to is a `PromptTarget`. Many attacks work generically with any `PromptTarget` including `RedTeamingOrchestrator` and `PromptSendingOrchestrator`.

With some algorithms, you want to send a prompt, set a system prompt, and modify conversation history (including PAIR, TAP, flip attack). These often require a `PromptChatTarget`, which implies you can modify a conversation history. `PromptChatTarget` is a subclass of `PromptTarget`.

Here are some examples:

| Example                             | Is `PromptChatTarget`?               | Notes                                                                                           |
|-------------------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------|
| **OpenAIChatTarget** (e.g., GPT-4)  | **Yes** (`PromptChatTarget`)         | Designed for conversational prompts (system messages, conversation history, etc.).               |
| **OpenAIDALLETarget**               | **No** (not a `PromptChatTarget`)    | Used for image generation; does not manage conversation history.                                 |
| **HTTPTarget**                      | **No** (not a `PromptChatTarget`)    | Generic HTTP target. Some apps might allow conversation history, but this target doesn't handle it. |
| **AzureBlobStorageTarget**          | **No** (not a `PromptChatTarget`)    | Used primarily for storage; not for conversation-based AI.                                       |
