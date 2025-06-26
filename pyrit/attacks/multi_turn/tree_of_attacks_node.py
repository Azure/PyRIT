# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List, Optional

from pyrit.exceptions import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestResponse, Score, SeedPrompt, SeedPromptGroup
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class TreeOfAttacksNode:
    """
    Represents a node in the Tree of Attacks with Pruning (TAP) strategy.

    Each node encapsulates an independent attack branch within the TAP algorithm's tree structure.
    Nodes manage their own conversation threads with both the adversarial chat target (for generating
    attack prompts) and the objective target (for testing those prompts). This design enables parallel
    exploration of multiple attack paths while maintaining conversation context isolation.

    The Tree of Attacks with Pruning strategy systematically explores a tree of possible attack paths,
    where each node represents a different approach or variation. The algorithm prunes less promising
    branches based on scoring results and explores the most successful paths more deeply.

    Node Lifecycle:
        1. Node is created with initial configuration and parent relationship
        2. `send_prompt_async()` executes one attack turn:
           - Generates an adversarial prompt using the red teaming chat
           - Optionally checks if the prompt is on-topic
           - Sends the prompt to the objective target
           - Scores the response to evaluate success
        3. Node can be duplicated to create child branches for further exploration
        4. Nodes track their execution state (completed, off_topic, scores)

    Note:
        `TreeOfAttacksNode` is typically not instantiated directly by users. Instead, it's created
        and managed internally by the `TreeOfAttacksWithPruningAttack` strategy during execution.
        The nodes form a tree structure where each branch represents a different attack approach,
        and the algorithm automatically prunes less successful branches while exploring promising ones.
    """

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_seed_prompt: SeedPrompt,
        adversarial_chat_prompt_template: SeedPrompt,
        adversarial_chat_system_seed_prompt: SeedPrompt,
        desired_response_prefix: str,
        objective_scorer: Scorer,
        on_topic_scorer: Optional[Scorer],
        request_converters: List[PromptConverterConfiguration],
        response_converters: List[PromptConverterConfiguration],
        auxiliary_scorers: Optional[List[Scorer]],
        attack_id: dict[str, str],
        memory_labels: Optional[dict[str, str]] = None,
        parent_id: Optional[str] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
    ) -> None:
        """
        Initialize a tree node.

        Args:
            objective_target (PromptChatTarget): The target to attack.
            adversarial_chat (PromptChatTarget): The chat target for generating adversarial prompts.
            adversarial_chat_seed_prompt (SeedPrompt): The seed prompt for the first turn.
            adversarial_chat_prompt_template (SeedPrompt): The template for subsequent turns.
            adversarial_chat_system_seed_prompt (SeedPrompt): The system prompt for the adversarial chat
            desired_response_prefix (str): The prefix for the desired response.
            objective_scorer (Scorer): The scorer for evaluating the objective target's response.
            on_topic_scorer (Optional[Scorer]): Optional scorer to check if the prompt is on-topic.
            request_converters (List[PromptConverterConfiguration]): Converters for request normalization
            response_converters (List[PromptConverterConfiguration]): Converters for response normalization
            auxiliary_scorers (Optional[List[Scorer]]): Additional scorers for the response
            attack_id (dict[str, str]): Unique identifier for the attack.
            memory_labels (Optional[dict[str, str]]): Labels for memory storage.
            parent_id (Optional[str]): ID of the parent node, if this is a child node
            prompt_normalizer (Optional[PromptNormalizer]): Normalizer for handling prompts and responses.
        """
        # Store configuration
        self._objective_target = objective_target
        self._adversarial_chat = adversarial_chat
        self._objective_scorer = objective_scorer
        self._adversarial_chat_seed_prompt = adversarial_chat_seed_prompt
        self._desired_response_prefix = desired_response_prefix
        self._adversarial_chat_prompt_template = adversarial_chat_prompt_template
        self._adversarial_chat_system_seed_prompt = adversarial_chat_system_seed_prompt
        self._on_topic_scorer = on_topic_scorer
        self._request_converters = request_converters
        self._response_converters = response_converters
        self._auxiliary_scorers = auxiliary_scorers or []
        self._attack_id = attack_id
        self._memory_labels = memory_labels or {}

        # Initialize utilities
        self._memory = CentralMemory.get_memory_instance()
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

        # Node identity
        self.parent_id = parent_id
        self.node_id = str(uuid.uuid4())

        # Conversation tracking
        self.objective_target_conversation_id = str(uuid.uuid4())
        self.adversarial_chat_conversation_id = str(uuid.uuid4())

        # Execution results (populated after send_prompt_async)
        self.completed = False
        self.off_topic = False
        self.objective_score: Optional[Score] = None
        self.auxiliary_scores: Dict[str, Score] = {}
        self.last_prompt_sent: Optional[str] = None
        self.last_response: Optional[str] = None
        self.error_message: Optional[str] = None

    async def send_prompt_async(self, objective: str) -> None:
        """
        Execute one turn of the attack for this node.

        This method orchestrates a complete attack iteration by generating an adversarial prompt,
        validating it, sending it to the target, and evaluating the response. The node's state
        is updated throughout the process to track execution progress and results.

        The method follows this workflow:
        1. Generate an adversarial prompt using the red teaming chat.
        2. Check if the prompt is on-topic (if configured).
        3. Send the prompt to the objective target.
        4. Score the response with all configured scorers.

        All errors are handled gracefully - JSON parsing errors and unexpected exceptions are
        caught and stored in the node's error_message attribute rather than being raised.

        Args:
            objective (str): The attack objective describing what the attacker wants to achieve.
                            This is used to guide the adversarial prompt generation and scoring.

        Returns:
            None: The method updates the node's internal state instead of returning values.
                Check node attributes like completed, off_topic, objective_score, and
                error_message to determine the execution outcome.

        Note:
            This method sets the following node attributes during execution:
            - `last_prompt_sent`: The generated adversarial prompt
            - `last_response`: The target's response
            - `objective_score`: The scoring result
            - `auxiliary_scores`: Additional scoring metrics
            - `completed`: `True` if execution finished successfully
            - `off_topic`: `True` if the prompt was deemed off-topic
            - `error_message`: Set if an error occurred during execution
        """

        try:
            # Generate adversarial prompt
            prompt = await self._generate_adversarial_prompt_async(objective)

            # Validate prompt is on-topic
            if await self._is_prompt_off_topic_async(prompt):
                return

            # Send prompt to objective target
            response = await self._send_prompt_to_target_async(prompt)

            # Score the response
            await self._score_response_async(response=response, objective=objective)

            # Mark execution as successful
            self._mark_execution_complete()

        except InvalidJsonException as e:
            self._handle_json_error(e)
        except Exception as e:
            self._handle_unexpected_error(e)

    async def _generate_adversarial_prompt_async(self, objective: str) -> str:
        """
        Generate an adversarial prompt using the red teaming chat.

        This method serves as the high-level interface for prompt generation, delegating
        to the more complex red teaming prompt generation that handles the actual
        communication with the adversarial chat target. It also updates the node's state
        to track the generated prompt.

        The generated prompt is designed to work towards the specified objective while
        attempting to bypass the target's safety mechanisms. The quality and approach
        of the prompt depends on the adversarial chat's capabilities and the configured
        system prompts.

        Args:
            objective (str): The attack objective describing what the attacker wants to achieve.
                            This objective is passed to the adversarial chat to guide the
                            generation of an appropriate attack prompt.

        Returns:
            str: The generated adversarial prompt text that will be sent to the objective
                target. This prompt is crafted to pursue the objective while attempting
                to avoid detection or refusal.

        Raises:
            InvalidJsonException: If the adversarial chat returns invalid JSON that cannot
                                be parsed to extract the prompt.
            RuntimeError: If the conversation history is in an unexpected state (e.g., no
                        assistant responses when expected).

        Side Effects:
            - Sets self.last_prompt_sent to the generated prompt
        """
        prompt = await self._generate_red_teaming_prompt_async(objective=objective)
        self.last_prompt_sent = prompt
        logger.debug(f"Node {self.node_id}: Generated adversarial prompt")
        return prompt

    async def _is_prompt_off_topic_async(self, prompt: str) -> bool:
        """
        Check if the generated prompt is off-topic using the on-topic scorer.

        This method evaluates whether the adversarial prompt aligns with the attack objective.
        Off-topic detection helps prune branches that have diverged from the intended goal,
        improving the efficiency of the tree exploration by focusing resources on relevant paths.

        The on-topic check is optional - if no on-topic scorer is configured, all prompts
        are considered on-topic by default. When a prompt is determined to be off-topic,
        the node is marked for pruning and will not be explored further.

        Args:
            prompt (str): The generated adversarial prompt to evaluate for topical relevance.

        Returns:
            bool: True if the prompt is off-topic (branch should be pruned), False if the
                prompt is on-topic or if no on-topic scorer is configured.

        Side Effects:
            - Sets self.off_topic to True if the prompt is determined to be off-topic

        Note:
            The on-topic scorer typically uses the attack objective to determine relevance.
            A prompt is considered off-topic if it asks for information that differs from
            or contradicts the original objective.
        """
        if not self._on_topic_scorer:
            return False

        on_topic_score = (await self._on_topic_scorer.score_text_async(text=prompt))[0]
        if not on_topic_score.get_value():
            logger.info(f"Node {self.node_id}: Generated prompt is off-topic, pruning branch")
            self.off_topic = True
            return True

        return False

    async def _send_prompt_to_target_async(self, prompt: str) -> PromptRequestResponse:
        """
        Send the generated adversarial prompt to the objective target.

        This method handles the communication with the target system, sending the attack prompt
        and retrieving the response. It uses the configured request and response converters to
        transform the prompt and response as needed (e.g., encoding variations, format changes).
        The prompt normalizer ensures consistent handling across different target types.

        The method creates a proper prompt structure, tracks the conversation context, and
        applies any configured labels and metadata before sending. This maintains the attack's
        conversation history for multi-turn scenarios.

        Args:
            prompt (str): The generated adversarial prompt to send to the target system.

        Returns:
            PromptRequestResponse: The response from the objective target, containing the
                                target's reply and associated metadata.

        Raises:
            ValueError: If no response is received from the target (e.g., connection failure).
            Exception: Any exceptions from the prompt normalizer or target communication.

        Side Effects:
            - Sets self.last_response to the target's response text
        """
        # Create seed prompt group from the generated prompt
        seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=prompt, data_type="text")])

        # Send prompt with configured converters
        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters,
            conversation_id=self.objective_target_conversation_id,
            target=self._objective_target,
            labels=self._memory_labels,
            orchestrator_identifier=self._attack_id,
        )

        # Store the last response text for reference
        response_piece = response.get_piece()
        self.last_response = response_piece.converted_value
        logger.debug(f"Node {self.node_id}: Received response from target")

        return response

    async def _score_response_async(self, *, response: PromptRequestResponse, objective: str) -> None:
        """
        Score the response from the objective target using the configured scorers.

        This method evaluates the target's response to determine how well it aligns with the
        attack objective. It applies both the primary objective scorer (which determines success)
        and any auxiliary scorers (which provide additional metrics). The scoring results are
        used by the TAP algorithm to decide which branches to explore further.

        The method leverages the Scorer utility to handle all scoring logic, including error
        handling and parallel execution of multiple scorers. Responses with errors are skipped
        to avoid scoring failures from blocking the attack progress.

        Args:
            response (PromptRequestResponse): The response from the objective target to evaluate.
                                            This contains the target's reply to the adversarial prompt.
            objective (str): The attack objective describing what the attacker wants to achieve.
                            This is passed to scorers as context for evaluation.

        Returns:
            None: The method updates the node's internal scoring state instead of returning values.

        Side Effects:
            - Sets self.objective_score to the primary scorer's result (if available)
            - Updates self.auxiliary_scores dictionary with results from auxiliary scorers

        Note:
            The objective score determines whether this branch achieved the attack goal.
            Higher scores indicate more successful attacks and influence which branches
            the TAP algorithm explores in subsequent iterations.
        """
        # Use the Scorer utility method to handle all scoring
        scoring_results = await Scorer.score_response_with_objective_async(
            response=response,
            auxiliary_scorers=self._auxiliary_scorers,
            objective_scorers=[self._objective_scorer],
            role_filter="assistant",
            task=objective,
            skip_on_error=True,
        )

        # Extract objective score
        objective_scores = scoring_results["objective_scores"]
        if objective_scores:
            self.objective_score = objective_scores[0]
            logger.debug(f"Node {self.node_id}: Objective score: {self.objective_score.get_value()}")

        # Extract auxiliary scores
        auxiliary_scores = scoring_results["auxiliary_scores"]
        for score in auxiliary_scores:
            scorer_name = score.scorer_class_identifier["__type__"]
            self.auxiliary_scores[scorer_name] = score
            logger.debug(f"Node {self.node_id}: {scorer_name} score: {score.get_value()}")

    def _mark_execution_complete(self) -> None:
        """
        Mark the node execution as successfully completed.

        This method updates the node's completion status and logs the final objective score.
        It should only be called after all attack steps (prompt generation, sending, and
        scoring) have finished successfully without errors. Nodes marked as complete are
        eligible for selection in the TAP algorithm's pruning and branching decisions.

        Side Effects:
            - Sets self.completed to True

        Note:
            This method is not called if the node encounters errors during execution
            or if the prompt is determined to be off-topic. In those cases, the node
            remains incomplete and may be pruned from further exploration.
        """
        self.completed = True
        score_str = self.objective_score.get_value() if self.objective_score else "N/A"
        logger.info(f"Node {self.node_id}: Completed with objective score {score_str}")

    def _handle_json_error(self, error: InvalidJsonException) -> None:
        """
        Handle JSON parsing errors from the adversarial chat.

        This method processes JSON-related errors that occur when parsing responses from the
        adversarial chat. Since the adversarial chat is expected to return structured JSON
        containing the attack prompt, parsing failures indicate the response format is invalid.
        The branch is pruned since it cannot proceed without a valid prompt.

        Args:
            error (InvalidJsonException): The JSON parsing exception that occurred during
                                        prompt generation or response parsing.

        Side Effects:
            - Sets self.error_message with a descriptive error message

        Note:
            When this error occurs, the node's execution is considered failed and the
            branch will be pruned from further exploration in the TAP algorithm.
        """
        logger.error(f"Node {self.node_id}: Failed to generate a prompt for the prompt target: {error}")
        logger.info("Pruning the branch since we can't proceed without red teaming prompt.")
        self.error_message = f"JSON parsing error: {str(error)}"

    def _handle_unexpected_error(self, error: Exception) -> None:
        """
        Handle unexpected errors during execution.

        This method serves as a catch-all error handler for any unanticipated exceptions
        that occur during the node's execution. It ensures the node fails gracefully
        without crashing the entire attack, allowing other branches to continue exploring.

        Args:
            error (Exception): The unexpected exception that occurred during any phase
                            of the node's execution.

        Side Effects:
            - Sets self.error_message with the error type and message

        Note:
            This handler ensures fault tolerance in the TAP algorithm. When one branch
            encounters an unexpected error, other branches can continue execution, making
            the attack more robust against transient failures or edge cases.
        """
        logger.error(f"Node {self.node_id}: Unexpected error during execution: {error}")
        self.error_message = f"Execution error: {str(error)}"

    def duplicate(self) -> TreeOfAttacksNode:
        """
        Create a duplicate of this node for branching.

        This method implements the branching mechanism of the TAP algorithm by creating
        a new node that inherits the current node's configuration and conversation history.
        The duplicate serves as a child node that can explore variations of the attack path
        while maintaining the context established by the parent.

        The duplication process preserves all configuration settings while creating new
        identifiers and duplicating conversation histories. This allows the child node to
        diverge from the parent's path while retaining the conversational context that
        led to the branching point.

        Returns:
            TreeOfAttacksNode: A new node instance that is a duplicate of this node,
                               ready to explore a new branch in the attack tree.

        Note:
            Duplication is a key operation in the TAP algorithm, enabling the exploration
            of multiple attack variations from promising nodes. The tree expands by
            duplicating successful nodes and pruning unsuccessful ones.
        """
        duplicate_node = TreeOfAttacksNode(
            objective_target=self._objective_target,
            adversarial_chat=self._adversarial_chat,
            adversarial_chat_seed_prompt=self._adversarial_chat_seed_prompt,
            adversarial_chat_prompt_template=self._adversarial_chat_prompt_template,
            adversarial_chat_system_seed_prompt=self._adversarial_chat_system_seed_prompt,
            objective_scorer=self._objective_scorer,
            on_topic_scorer=self._on_topic_scorer,
            request_converters=self._request_converters,
            response_converters=self._response_converters,
            auxiliary_scorers=self._auxiliary_scorers,
            attack_id=self._attack_id,
            memory_labels=self._memory_labels,
            desired_response_prefix=self._desired_response_prefix,
            parent_id=self.node_id,
            prompt_normalizer=self._prompt_normalizer,
        )

        # Duplicate the conversations to preserve history
        duplicate_node.objective_target_conversation_id = self._memory.duplicate_conversation(
            conversation_id=self.objective_target_conversation_id
        )

        duplicate_node.adversarial_chat_conversation_id = self._memory.duplicate_conversation(
            conversation_id=self.adversarial_chat_conversation_id
        )

        logger.debug(f"Node {self.node_id}: Created duplicate node {duplicate_node.node_id}")

        return duplicate_node

    @pyrit_json_retry
    async def _generate_red_teaming_prompt_async(self, objective: str) -> str:
        """
        Generate an adversarial prompt using the red teaming chat.

        This method handles the core logic of prompt generation by communicating with the
        adversarial chat target. It adapts its approach based on whether this is the first
        turn (using a seed prompt) or a subsequent turn (using conversation history and scores).
        The red teaming chat returns a structured JSON response containing the attack prompt.

        The method follows different strategies:
        - First turn: Initializes the system prompt and uses the seed prompt template
        - Subsequent turns: Uses conversation history and previous scores to guide generation

        Args:
            objective (str): The attack objective describing what the attacker wants to achieve.
                            This guides both the system prompt configuration and prompt generation.

        Returns:
            str: The generated adversarial prompt text extracted from the JSON response.

        Raises:
            InvalidJsonException: If the adversarial chat response cannot be parsed as JSON
                                or lacks required fields.
            RuntimeError: If the conversation history is in an unexpected state (e.g., no
                        assistant responses found when expected in subsequent turns).
        """
        # Check if this is the first turn or subsequent turn
        if self._is_first_turn():
            prompt_text = await self._generate_first_turn_prompt_async(objective)
        else:
            prompt_text = await self._generate_subsequent_turn_prompt_async(objective)

        # Send to adversarial chat and get JSON response
        adversarial_response = await self._send_to_adversarial_chat_async(prompt_text)

        # Parse and return the prompt from the response
        return self._parse_red_teaming_response(adversarial_response)

    def _is_first_turn(self) -> bool:
        """
        Check if this is the first turn of the conversation.

        This method determines whether the node is executing its initial attack turn by
        examining the objective target conversation history.

        Returns:
            bool: True if no messages exist in the objective target conversation (first turn),
                False if the conversation already contains messages (subsequent turns).
        """
        target_messages = self._memory.get_conversation(conversation_id=self.objective_target_conversation_id)
        return not target_messages

    async def _generate_first_turn_prompt_async(self, objective: str) -> str:
        """
        Generate the prompt for the first turn using the seed prompt.

        This method handles the special initialization required for the first attack turn.
        It sets up the adversarial chat's system prompt to establish the attack context and
        returns a seed prompt to begin the conversation. The system prompt configures the
        adversarial chat's behavior for all subsequent interactions, while the seed prompt
        provides the initial query to start generating attack prompts.

        The first turn is unique because there's no conversation history to build upon,
        so the method uses predefined templates that are designed to initiate the attack
        sequence effectively.

        Args:
            objective (str): The attack objective used to customize both the system prompt
                            and seed prompt.

        Returns:
            str: The rendered seed prompt text that will be sent to the adversarial chat
                to generate the first attack prompt.
        """
        # Initialize system prompt for adversarial chat
        system_prompt = self._adversarial_chat_system_seed_prompt.render_template_value(
            objective=objective, desired_prefix=self._desired_response_prefix
        )

        self._adversarial_chat.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=self.adversarial_chat_conversation_id,
            orchestrator_identifier=self._attack_id,
            labels=self._memory_labels,
        )

        logger.debug(f"Node {self.node_id}: Using initial seed prompt for first turn")

        # Use seed prompt for first turn
        return self._adversarial_chat_seed_prompt.render_template_value(objective=objective)

    async def _generate_subsequent_turn_prompt_async(self, objective: str) -> str:
        """
        Generate the prompt for subsequent turns using the template.

        This method creates prompts for all turns after the first by incorporating conversation
        history and previous scoring results. It retrieves the target's last response and its
        associated score, then uses the prompt template to generate a context-aware prompt that
        builds upon the established conversation. This approach allows the adversarial chat to
        adapt its strategy based on what has worked or failed in previous attempts.

        The method ensures continuity in the attack by providing the adversarial chat with
        feedback about the target's responses and their effectiveness, enabling more
        sophisticated multi-turn attack strategies.

        Args:
            objective (str): The attack objective that guides the prompt generation and
                            provides context for the adversarial chat.

        Returns:
            str: The rendered prompt text containing the target's last response, the objective,
                and the score.

        Raises:
            RuntimeError: If no assistant responses are found in the conversation history.
                        This indicates a broken conversation state since subsequent turns
                        require at least one prior exchange.
        """
        # Get conversation history
        target_messages = self._memory.get_conversation(conversation_id=self.objective_target_conversation_id)

        # Extract the last assistant response
        assistant_responses = [r for r in target_messages if r.get_piece().role == "assistant"]
        if not assistant_responses:
            logger.error(f"No assistant responses found in the conversation {self.objective_target_conversation_id}.")
            raise RuntimeError("Cannot proceed without an assistant response.")

        target_response = assistant_responses[-1]
        target_response_piece = target_response.get_piece()
        logger.debug(f"Node {self.node_id}: Using response {target_response_piece.id} for next prompt")

        # Get score for the response
        score = await self._get_response_score_async(str(target_response_piece.id))

        # Generate prompt using template
        return self._adversarial_chat_prompt_template.render_template_value(
            target_response=target_response_piece.converted_value,
            objective=objective,
            score=str(score),
        )

    async def _get_response_score_async(self, response_id: str) -> str:
        """
        Get the score for a response from memory.

        This method retrieves the scoring result for a previous response from the memory store.
        It's used during subsequent turn prompt generation to provide the adversarial chat with
        feedback about how well previous attempts achieved the objective. The score helps the
        adversarial chat adjust its strategy for generating more effective prompts.

        Args:
            response_id (str): The unique identifier of the response to retrieve the score for.

        Returns:
            str: The score value as a string representation. Returns "unavailable" if no score
                exists for the given response ID. For numeric scores, this will be the string
                representation of the float value (e.g., "0.75").

        Note:
            The method assumes that if scores exist, at least one score will be present in the
            list. It takes the first score if multiple scores are associated with the response,
            which is typically the objective score in the TAP algorithm context.
        """
        scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(response_id)])
        return str(scores[0].get_value()) if scores else "unavailable"

    async def _send_to_adversarial_chat_async(self, prompt_text: str) -> str:
        """
        Send a prompt to the adversarial chat and get the response.

        This method handles the low-level communication with the adversarial chat target.
        It configures the request to expect a JSON response format, packages the prompt
        appropriately, and manages the conversation context. The adversarial chat is expected
        to return structured JSON containing the generated attack prompt and related metadata.

        The method uses the prompt normalizer to ensure consistent communication patterns
        and maintains the conversation history in the adversarial chat thread, separate from
        the objective target conversation.

        Args:
            prompt_text (str): The text to send to the adversarial chat. This could be either
                            the initial seed prompt or a template-generated prompt containing
                            conversation history and scores.

        Returns:
            str: The raw response from the adversarial chat, expected to be JSON formatted.
                This response should contain at least a "prompt" field with the generated
                attack prompt.
        """
        # Configure for JSON response
        prompt_metadata: dict[str, str | int] = {"response_format": "json"}
        seed_prompt_group = SeedPromptGroup(
            prompts=[SeedPrompt(value=prompt_text, data_type="text", metadata=prompt_metadata)]
        )

        # Send and get response
        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            conversation_id=self.adversarial_chat_conversation_id,
            target=self._adversarial_chat,
            labels=self._memory_labels,
            orchestrator_identifier=self._attack_id,
        )

        return response.get_value()

    def _parse_red_teaming_response(self, red_teaming_response: str) -> str:
        """
        Extract the prompt field from JSON response.

        This method parses the structured response from the adversarial chat to extract
        the generated attack prompt. The adversarial chat is expected to return JSON with
        at least a "prompt" field containing the attack text. The method handles common
        formatting issues like markdown wrappers that LLMs sometimes add around JSON.

        The parsing is strict - the response must be valid JSON and must contain the
        required "prompt" field. This ensures the TAP algorithm receives well-formed
        prompts for attacking the objective target.

        Args:
            red_teaming_response (str): The raw response from the red teaming chat, expected
                                    to be JSON formatted (possibly wrapped in markdown).
                                    Should contain at least {"prompt": "attack text"}.

        Returns:
            str: The prompt extracted from the JSON response. This is the actual attack
                text that will be sent to the objective target.

        Raises:
            InvalidJsonException: If the response is not valid JSON after removing markdown
                                formatting, or if the parsed JSON does not contain a "prompt"
                                field.
        """
        # Remove markdown formatting if present
        red_teaming_response = remove_markdown_json(red_teaming_response)

        try:
            red_teaming_response_dict = json.loads(red_teaming_response)
        except json.JSONDecodeError:
            logger.error(f"The response from the red teaming chat is not in JSON format: {red_teaming_response}")
            raise InvalidJsonException(message="The response from the red teaming chat is not in JSON format.")

        try:
            return red_teaming_response_dict["prompt"]
        except KeyError:
            logger.error(f"The response from the red teaming chat does not contain a prompt: {red_teaming_response}")
            raise InvalidJsonException(message="The response from the red teaming chat does not contain a prompt.")

    def __str__(self) -> str:
        """String representation of the node showing key execution results."""
        return (
            "TreeOfAttackNode("
            f"completed={self.completed}, "
            f"objective_score={self.objective_score.get_value() if self.objective_score else None}, "
            f"node_id={self.node_id}, "
            f"objective_target_conversation_id={self.objective_target_conversation_id})"
        )

    __repr__ = __str__
