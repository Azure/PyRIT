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
    Represents a node in the Tree of Attacks with Pruning strategy.

    Each node manages its own conversation threads with both the adversarial
    chat (for generating prompts) and the objective target (for testing prompts).

    The node lifecycle:
    1. Node is created with initial configuration
    2. `send_prompt_async` is called to execute one attack turn
    3. The node generates an adversarial prompt
    4. Optionally checks if the prompt is on-topic
    5. Sends the prompt to the objective target
    6. Scores the response
    7. Node can be duplicated for branching
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
        self.prompt_sent = False
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

        This method orchestrates the complete attack turn:
        1. Generate an adversarial prompt using the red teaming chat
        2. Check if the prompt is on-topic (if configured)
        3. Send the prompt to the objective target
        4. Score the response with all configured scorers

        The method handles errors gracefully and updates the node's state
        throughout the execution process.

        Args:
            objective (str): The attack objective describing what the attacker wants to achieve.
        """
        self.prompt_sent = True

        try:
            # Step 1: Generate adversarial prompt
            prompt = await self._generate_adversarial_prompt_async(objective)

            # Step 2: Validate prompt is on-topic
            if await self._is_prompt_off_topic_async(prompt):
                return

            # Step 3: Send prompt to objective target
            response = await self._send_prompt_to_target_async(prompt)

            # Step 4: Score the response
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

        Args:
            objective (str): The attack objective describing what the attacker wants to achieve.

        Returns:
            str: The generated adversarial prompt.
        """
        prompt = await self._generate_red_teaming_prompt_async(objective=objective)
        self.last_prompt_sent = prompt
        logger.debug(f"Node {self.node_id}: Generated adversarial prompt")
        return prompt

    async def _is_prompt_off_topic_async(self, prompt: str) -> bool:
        """
        Check if the generated prompt is off-topic using the on-topic scorer.

        Args:
            prompt (str): The prompt to check.

        Returns:
            bool: True if the prompt is off-topic, False otherwise.
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
        This method sends the generated prompt to the objective target and
        waits for the response. It uses the configured request and response
        converters to normalize the prompt and response.

        Args:
            prompt (str): The generated adversarial prompt to send.

        Returns:
            PromptRequestResponse: The response from the objective target after sending the prompt.
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
        This method uses the Scorer utility to handle all scoring logic, including
        objective and auxiliary scorers.

        Args:
            response (PromptRequestResponse): The response from the objective target.
            objective (str): The attack objective describing what the attacker wants to achieve.
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
        """Mark the node execution as successfully completed."""
        self.completed = True
        score_str = self.objective_score.get_value() if self.objective_score else "N/A"
        logger.info(f"Node {self.node_id}: Completed with objective score {score_str}")

    def _handle_json_error(self, error: InvalidJsonException) -> None:
        """
        Handle JSON parsing errors from the adversarial chat.

        Args:
            error (InvalidJsonException): The InvalidJsonException that occurred during prompt generation.
        """
        logger.error(f"Node {self.node_id}: Failed to generate a prompt for the prompt target: {error}")
        logger.info("Pruning the branch since we can't proceed without red teaming prompt.")
        self.error_message = f"JSON parsing error: {str(error)}"

    def _handle_unexpected_error(self, error: Exception) -> None:
        """
        Handle unexpected errors during execution.

        Args:
            error (Exception): The unexpected exception that occurred.
        """
        logger.error(f"Node {self.node_id}: Unexpected error during execution: {error}")
        self.error_message = f"Execution error: {str(error)}"

    def duplicate(self) -> TreeOfAttacksNode:
        """
        Create a duplicate of this node for branching.

        The duplicate inherits:
        - All configuration from the parent node
        - Full conversation history (duplicated in memory)
        - Parent-child relationship (this node becomes the parent)

        The duplicate gets new:
        - Node ID
        - Conversation IDs (but with duplicated history)

        Returns:
            TreeOfAttacksNode: A new node instance with the same configuration and duplicated conversations.
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

        This method handles two scenarios:
        1. First turn: Uses the seed prompt and initializes the system prompt
        2. Subsequent turns: Uses the template with conversation history and scores

        The red teaming chat is expected to return a JSON response with a "prompt" field.

        Args:
            objective (str): The attack objective describing what the attacker wants to achieve.

        Returns:
            str: The generated adversarial prompt text.

        Raises:
            InvalidJsonException: If the response cannot be parsed as JSON.
            RuntimeError: If no assistant response is found when expected.
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

        Returns:
            True if no messages exist in the objective target conversation.
        """
        target_messages = self._memory.get_conversation(conversation_id=self.objective_target_conversation_id)
        return not target_messages

    async def _generate_first_turn_prompt_async(self, objective: str) -> str:
        """
        Generate the prompt for the first turn using the seed prompt.

        Also initializes the system prompt for the adversarial chat.

        Args:
            objective (str): The attack objective.

        Returns:
            str: The rendered seed prompt text for the first turn.
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
        Uses the conversation history and previous scores to inform the next prompt.

        Args:
            objective (str): The attack objective.

        Returns:
            str: The rendered prompt text for the next turn.

        Raises:
            RuntimeError: If no assistant response is found in the conversation history.
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

        Args:
            response_id (str): The ID of the response to get the score for.

        Returns:
            str: The score value as a string, or "unavailable" if no score exists.
        """
        scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(response_id)])
        return str(scores[0].get_value()) if scores else "unavailable"

    async def _send_to_adversarial_chat_async(self, prompt_text: str) -> str:
        """
        Send a prompt to the adversarial chat and get the response.

        Args:
            prompt_text (str): The text of the prompt to send.

        Returns:
            str: The raw response from the adversarial chat, expected to be JSON formatted.
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

        Args:
            red_teaming_response (str): The raw response from the red teaming chat, expected to be JSON formatted.

        Returns:
            str: The prompt extracted from the response.

        Raises:
            InvalidJsonException: If the response is not valid JSON or does not contain a "prompt" field.
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
