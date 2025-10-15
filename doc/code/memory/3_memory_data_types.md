# 3. Memory Types

There are several types of data you can retrieve from memory at any point in time using the `MemoryInterface`.

## PromptRequestPiece and PromptRequestResponse

One of the most fundamental data structures in PyRIT is [PromptRequestPiece](../../../pyrit/models/prompt_request_piece.py) and [PromptRequestResponse](../../../pyrit/models/prompt_request_response.py). These classes provide the foundation for multi-modal interaction tracking throughout the framework.

### PromptRequestPiece

`PromptRequestPiece` represents a single piece of a request to a target. It is the atomic unit that is stored in the database and contains comprehensive metadata about each interaction.

**Key Fields:**

- **`id`**: Unique identifier for the piece (UUID)
- **`conversation_id`**: Identifier grouping pieces into a single conversation with a target
- **`sequence`**: Order of the piece within a conversation
- **`role`**: The role in the conversation (e.g., `user`, `assistant`, `system`)
- **`original_value`**: The original prompt text or file path (for images, audio, etc.)
- **`original_value_data_type`**: The data type of the original value (e.g., `text`, `image_path`, `audio_path`)
- **`converted_value`**: The prompt after any conversions/transformations have been applied
- **`converted_value_data_type`**: The data type of the converted value
- **`labels`**: Dictionary of labels for categorization and filtering
- **`prompt_metadata`**: Component-specific metadata (e.g., blob URIs, document types)
- **`converter_identifiers`**: List of converters applied to transform the prompt
- **`prompt_target_identifier`**: Information about the target that received this prompt
- **`attack_identifier`**: Information about the attack that generated this prompt
- **`scorer_identifier`**: Information about the scorer that evaluated this prompt
- **`response_error`**: Error status (e.g., `none`, `blocked`, `processing`)
- **`originator`**: Source of the prompt (`attack`, `converter`, `scorer`, `undefined`)
- **`scores`**: List of `Score` objects associated with this piece
- **`targeted_harm_categories`**: Harm categories associated with the prompt
- **`timestamp`**: When the piece was created

This rich context allows PyRIT to track the full lifecycle of each interaction, including transformations, targeting, scoring, and error handling.

### PromptRequestResponse

`PromptRequestResponse` represents a single request or response to a target and can contain multiple `PromptRequestPieces`. This allows for multi-modal interactions where, for example, you send both text and an image in a single request.

**Examples:**
- A text-only message: 1 `PromptRequestResponse` containing 1 `PromptRequestPiece`
- An image with caption: 1 `PromptRequestResponse` containing 2 `PromptRequestPieces` (text + image)
- A conversation: Multiple `PromptRequestResponses` linked by the same `conversation_id`

**Validation Rules:**
- All `PromptRequestPieces` in a `PromptRequestResponse` must share the same
   -  `conversation_id`
   - `sequence` number
   - `role`
- All `PromptRequestPieces` have a non-null `converted_value`

### Conversation Structure

A conversation is a list of `PromptRequestResponses` that share the same `conversation_id`. The sequence of the `PromptRequestPieces` and their corresponding `PromptRequestResponses` dictates the order of the conversation.

Here is a sample conversation made up of three `PromptRequestResponses` which all share the same conversation ID. The first `PromptRequestResponse` in the image contains two parts—a text `PromptRequestPiece` and an image `PromptRequestPiece`.

![PromptRequestPiece and PromptRequestResponse architecture](../../../assets/prompt_request_piece.png)

This architecture is plumbed throughout PyRIT, providing flexibility to interact with various modalities seamlessly. All pieces are stored in the database as individual `PromptRequestPieces` and are reassembled when needed. The `PromptNormalizer` automatically adds these to the database as prompts are sent.

## SeedPrompts

[`SeedPrompt`](../../../pyrit/models/seed_prompt.py) objects represent the starting points of conversations or attack objectives. They are used to assemble and initiate attacks, and can be translated to and from `PromptRequestPieces`.

**Key Fields:**

- **`value`**: The actual prompt text or file path
- **`data_type`**: Type of data (e.g., `text`, `image_path`, `audio_path`)
- **`name`**: Name of the prompt
- **`dataset_name`**: Name of the dataset this prompt belongs to
- **`harm_categories`**: Categories of harm associated with this prompt
- **`description`**: Description of the prompt's purpose or content
- **`parameters`**: Template parameters that can be filled in dynamically
- **`prompt_group_id`**: Groups related prompts together
- **`role`**: Role in conversation (e.g., `user`, `assistant`)
- **`metadata`**: Arbitrary metadata that can be attached

`SeedPrompts` support Jinja2 templating, allowing dynamic prompt generation with parameter substitution. They can be loaded from YAML files and organized into datasets and groups for systematic testing.

## Scores

[`Score`](../../../pyrit/models/score.py) objects represent evaluations of prompts or responses. Scores are generated by scorer components and attached to `PromptRequestPieces` to track the success or characteristics of attacks. When a prompt is scored, it is added to the database and can be queried later.

**Key Fields:**

- **`score_value`**: The actual score (e.g., `"true"`, `"0.75"`)
- **`score_value_description`**: Human-readable description of the score
- **`score_type`**: Type of score (`true_false` or `float_scale`)
- **`score_category`**: Categories the score applies to (e.g., `["hate", "violence"]`)
- **`score_rationale`**: Explanation of why the score was assigned
- **`scorer_class_identifier`**: Information about the scorer that generated this score
- **`prompt_request_response_id`**: The ID of the piece/response being scored
- **`task`**: The original attacker's objective being evaluated
- **`score_metadata`**: Custom metadata specific to the scorer

Scores enable automated evaluation of attack success, content harmfulness, and other metrics throughout PyRIT's red teaming workflows.

## AttackResults

[`AttackResult`](../../../pyrit/models/attack_result.py) objects encapsulate the complete outcome of an attack execution, including metrics, evidence, and success determination. When an attack is run, the AttackResult is added to the database and can be queried later.

**Key Fields:**

- **`conversation_id`**: The conversation that produced this result
- **`objective`**: Natural-language description of the attacker's goal
- **`attack_identifier`**: Information identifying the attack strategy used
- **`last_response`**: The final `PromptRequestPiece` generated in the attack
- **`last_score`**: The final score assigned to the last response
- **`executed_turns`**: Number of turns executed in the attack
- **`execution_time_ms`**: Total execution time in milliseconds
- **`outcome`**: The attack outcome (`SUCCESS`, `FAILURE`, or `UNDETERMINED`)
- **`outcome_reason`**: Optional explanation for the outcome
- **`related_conversations`**: Set of related conversation references
- **`metadata`**: Arbitrary metadata about the attack execution

`AttackResult` objects provide comprehensive reporting on attack campaigns, enabling analysis of red teaming effectiveness and vulnerability identification.
