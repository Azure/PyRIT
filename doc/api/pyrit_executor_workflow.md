# pyrit.executor.workflow

Workflow components and strategies used by the PyRIT executor.

## `class XPIAContext(WorkflowContext)`

Context for Cross-Domain Prompt Injection Attack (XPIA) workflow.

Contains execution-specific parameters needed for each XPIA attack run.
Immutable objects like targets and scorers are stored in the workflow instance.

## `class XPIAManualProcessingWorkflow(XPIAWorkflow)`

XPIA workflow with manual processing intervention.

This variant pauses execution to allow manual triggering of the
processing target, then accepts the output via console input.
This is useful for scenarios where the processing target requires
manual interaction or cannot be automated.

The workflow will prompt the operator to manually trigger the processing
target's execution and paste the output into the console for scoring.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `attack_setup_target` | `PromptTarget` | The target that generates the attack prompt and gets it into the attack location. |
| `scorer` | `Scorer` | The scorer to use to score the processing response. This is required to evaluate the manually provided response. |
| `converter_config` | `Optional[StrategyConverterConfig]` | Optional converter configuration for request and response converters. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Optional PromptNormalizer instance. If not provided, a new one will be created. Defaults to `None`. |
| `logger` | `logging.Logger` | Logger instance for logging events. Defaults to `logger`. |

## `class XPIAProcessingCallback(Protocol)`

Protocol for processing callback functions used in XPIA workflows.

Defines the interface for callback functions that execute the processing
phase of an XPIA attack. The callback should handle the actual execution
of the processing target and return the response as a string.

## `class XPIAResult(WorkflowResult)`

Result of XPIA workflow execution.

Contains the outcome of the cross-domain prompt injection attack, including
the processing response, optional score, and attack setup response.

## `class XPIAStatus(Enum)`

Enumeration of possible XPIA attack result statuses.

## `class XPIATestWorkflow(XPIAWorkflow)`

XPIA workflow with automated test processing.

This variant automatically handles the processing phase by sending
a predefined prompt to a processing target. It is designed for automated
testing scenarios where the processing can be scripted rather than manual.

The workflow creates an automated processing callback that sends the
processing prompt to the configured processing target and returns the response.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `attack_setup_target` | `PromptTarget` | The target that generates the attack prompt and gets it into the attack location. |
| `processing_target` | `PromptTarget` | The target of the attack which processes the processing prompt. This should include references to invoke plugins (if any). |
| `scorer` | `Scorer` | The scorer to use to score the processing response. This is required for test workflows to evaluate attack success. |
| `converter_config` | `Optional[StrategyConverterConfig]` | Optional converter configuration for request and response converters. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Optional PromptNormalizer instance. If not provided, a new one will be created. Defaults to `None`. |
| `logger` | `logging.Logger` | Logger instance for logging events. Defaults to `logger`. |

## `class XPIAWorkflow(WorkflowStrategy[XPIAContext, XPIAResult], Identifiable)`

Implementation of Cross-Domain Prompt Injection Attack (XPIA) workflow.

This workflow orchestrates an attack where:
1. An attack prompt is generated and positioned using the attack_setup_target
2. The processing_callback is executed to trigger the target's processing
3. The response is optionally scored to determine success

The workflow supports customization through prompt converters and scorers,
allowing for various attack techniques and evaluation methods.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `attack_setup_target` | `PromptTarget` | The target that generates the attack prompt and gets it into the attack location. |
| `scorer` | `Optional[Scorer]` | Optional scorer to evaluate the processing response. If no scorer is provided the workflow will skip scoring. Defaults to `None`. |
| `converter_config` | `Optional[StrategyConverterConfig]` | Optional converter configuration for request and response converters. Defaults to `None`. |
| `prompt_normalizer` | `Optional[PromptNormalizer]` | Optional PromptNormalizer instance. If not provided, a new one will be created. Defaults to `None`. |
| `logger` | `logging.Logger` | Logger instance for logging events. Defaults to `logger`. |

**Methods:**

#### `execute_async(kwargs: Any = {}) → XPIAResult`

Execute the XPIA workflow strategy asynchronously with the provided parameters.

| Parameter | Type | Description |
|---|---|---|
| `attack_content` | `Message` | The content to use for the attack. |
| `processing_callback` | `ProcessingCallback` | The callback to execute after the attack prompt is positioned in the attack location. This is generic on purpose to allow for flexibility. The callback should return the processing response. |
| `processing_prompt` | `Optional[Message]` | The prompt to send to the processing target. This should include placeholders to invoke plugins (if any). |
| `memory_labels` | `Optional[Dict[str, str]]` | Memory labels for the attack context. |
| `**kwargs` | `Any` | Additional parameters for the attack. Defaults to `{}`. |

**Returns:**

- `XPIAResult` — The result of the workflow execution.

**Raises:**

- `TypeError` — If any of the provided parameters are of incorrect type.
