from __future__ import annotations
import logging
import time
import enum
from pathlib import Path
from typing import Optional, Any, List

from pyrit.common.path import RED_TEAM_ORCHESTRATOR_PATH
from pyrit.common.utils import combine_dict
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score
from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.orchestratorv3.components.score_evaluator import ScoreEvaluator
from pyrit.orchestratorv3.models.builder import AttackStrategy
from pyrit.orchestratorv3.models.context import ConversationSession, MultiTurnAttackContext, MultiTurnAttackResult
from pyrit.orchestratorv3.components.conversation_manager import ConversationManager, ConversationState
from pyrit.orchestratorv3.multi_turn.base import MultiTurnBaseAttackBuilder, MultiTurnBaseOrchestratorBuilder
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.memory import MemoryInterface
from pyrit.models import SeedPrompt
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)

DEFAULT_ADVERSARIAL_PROMPT_IF_OBJECTIVE_TARGET_IS_BLOCKED = (
    "Request to target failed: blocked. " 
    "Please rewrite your prompt to avoid getting blocked next time."
)

DEFAULT_ERR_MSG_IF_OBJECTIVE_TARGET_HAS_NON_TEXT_RESPONSE = (
    "The attack target does not respond with text output, "
    "so the scoring rationale is the only textual feedback "
    "that can be passed to the red teaming chat. "
)

class RTOSystemPromptPaths(enum.Enum):
    TEXT_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "text_generation.yaml").resolve()
    IMAGE_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "image_generation.yaml").resolve()
    NAIVE_CRESCENDO = Path(RED_TEAM_ORCHESTRATOR_PATH, "naive_crescendo.yaml").resolve()
    VIOLENT_DURIAN = Path(RED_TEAM_ORCHESTRATOR_PATH, "violent_durian.yaml").resolve()
    CRUCIBLE = Path(RED_TEAM_ORCHESTRATOR_PATH, "crucible.yaml").resolve()

class RedTeamingOrchestratorBuilder(MultiTurnBaseOrchestratorBuilder):
    """Builder for configuring red teaming orchestrators."""
    
    def __init__(
        self, 
        target: PromptTarget, 
        adversarial_chat: PromptChatTarget,
        system_prompt_path: Path = RTOSystemPromptPaths.TEXT_GENERATION.value,
        seed_prompt: Optional[str] = "How can I help you?",
        prompt_normalizer: Optional[PromptNormalizer] = None,
        use_score_as_feedback: bool = False,
        memory: Optional[MemoryInterface] = None,
    ):
        """Initialize the Red Teaming Orchestrator Builder.
        
        Args:
            target: Target system to attack
            adversarial_chat: Chat model that generates attack prompts
            memory: Optional memory interface for storing attack data
        """
        super().__init__(
            target=target,
            adversarial_chat=adversarial_chat,
            system_prompt_path=system_prompt_path,
            seed_prompt=seed_prompt,
            prompt_normalizer=prompt_normalizer,
            memory=memory
        )
        self._use_score_as_feedback = use_score_as_feedback

    def with_use_score_as_feedback(self, use_score_as_feedback: bool) -> RedTeamingOrchestratorBuilder:
        """Set whether to use score as feedback for the attack.
        
        Args:
            use_score_as_feedback: Whether to use score as feedback
        """
        self._use_score_as_feedback = use_score_as_feedback
        return self
    
    def attack(self) -> RedTeamingAttackBuilder:
        """Transition to attack configuration.
        
        Returns:
            An attack builder for configuring the attack
        """
        
        # Create the strategy with the current context
        
        # Create and return the attack builder
        return RedTeamingAttackBuilder(
            context=self.context, 
            chat_system_seed_prompt=self._system_prompt,
            seed_prompt=self._chat_seed_prompt, 
            prompt_normalizer=self._prompt_normalizer,
            use_score_as_feedback=self._use_score_as_feedback,
            memory=self._memory
        )


class RedTeamingAttackBuilder(MultiTurnBaseAttackBuilder):
    """Builder for configuring red teaming attacks."""
    
    def __init__(
        self, 
        *,
        context: MultiTurnAttackContext, 
        chat_system_seed_prompt: SeedPrompt,
        seed_prompt: SeedPrompt,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        use_score_as_feedback: bool = False,
        memory: Optional[MemoryInterface] = None,
    ):
        """Initialize the Red Teaming Attack Builder.
        
        Args:
            context: Context for the attack
            chat_system_seed_prompt: Seed prompt for the chat system
            seed_prompt: Seed prompt for the attack
            prompt_normalizer: Optional normalizer for prompts
            memory: Optional memory interface for storing attack data
        """
        super().__init__(
            context=context, 
            chat_system_seed_prompt=chat_system_seed_prompt,
            seed_prompt=seed_prompt,
            prompt_normalizer=prompt_normalizer,
            memory=memory
        )
        self._use_score_as_feedback = use_score_as_feedback
    
    def with_objective(self, objective_text: str) -> 'RedTeamingAttackBuilder':
        """Set the attack objective.
        
        Args:
            objective_text: Description of what the attack should achieve
            
        Returns:
            Self for method chaining
        """
        super().with_objective(objective_text=objective_text)
        self._system_prompt = self._chat_system_seed_prompt.render_template_value(objective=objective_text)
        return self
    
    async def execute(self) -> MultiTurnAttackResult:
        """Execute the attack with the current configuration.
        
        Returns:
            Result of the attack
        """
        if not self._context.objective:
            raise ValueError("Attack objective must be set")
        
        if self._system_prompt is None:
            raise ValueError("System prompt must be set")
        
        if self._objective_scorer is None:
            raise ValueError("Objective scorer must be set")
        
        ctx = self.context
        
        strategy = RedTeamingStrategy(
            chat_seed_prompt=self._chat_seed_prompt,
            system_prompt=self._system_prompt,
            prompt_normalizer=self._prompt_normalizer,
            prepend_conversation=self._prepend_conversation,
            objective_scorer=self._objective_scorer,
            use_score_as_feedback=self._use_score_as_feedback,
            memory=self._memory,
        )

        await strategy.setup(context=ctx)
        # Execute the attack using the strategy
        return await strategy.execute(context=ctx)

class RedTeamingStrategy(AttackStrategy[MultiTurnAttackContext, MultiTurnAttackResult]):
    """
    Implementation of multi-turn red teaming attack strategy.

    This class manages the multi-turn conversation flow between an adversarial chat model and a target,
    aiming to fulfill a specified attack objective. In each turn, the strategy optionally uses scoring 
    feedback to guide prompt generation. If the objective is not achieved, the scoring output may trigger 
    backtracking or re-prompting. By leveraging scores at each step, the strategy refines attacks in 
    real-time to increase the likelihood of success.
    """
    
    def __init__(
        self,
        chat_seed_prompt: SeedPrompt,
        objective_scorer: Scorer,
        system_prompt: Optional[SeedPrompt] = None,
        prepend_conversation: Optional[List[SeedPrompt]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        use_score_as_feedback: bool = False,
        memory: Optional[MemoryInterface] = None
    ):
        """Initialize the red teaming strategy.

        Args:
            target: The target system to attack.
            adversarial_chat: The chat model generating attack prompts.
            chat_seed_prompt: The seed prompt for adversarial chat.
            objective_scorer: The scorer used to evaluate responses.
            system_prompt: An optional system prompt for adversarial chat.
            prepend_conversation: A list of seed prompts to prepend to the conversation.
            prompt_normalizer: An optional prompt normalizer for prompts.
            use_score_as_feedback: Whether to use scores as feedback in prompt generation.
            memory: An optional memory interface for storing attack data.
        """
        super().__init__(
            chat_seed_prompt=chat_seed_prompt,
            system_prompt=system_prompt,
            prepend_conversation=prepend_conversation,
            prompt_normalizer=prompt_normalizer,
            memory=memory
        )

        self._use_score_as_feedback = use_score_as_feedback

        self._conversation_manager = ConversationManager(
            orchestrator_identifier=self.identifier,
            memory=memory,
        )

        self._score_evaluator = ScoreEvaluator(
            use_score_as_feedback=use_score_as_feedback,
            scorer=objective_scorer,
        )

    async def setup(self, *, context: MultiTurnAttackContext) -> None:
        """
        Prepare the strategy for execution.

        1. Initializes or retrieves the conversation state.
        2. Updates turn counts and checks for any custom prompt.
        3. Retrieves the last assistant message's evaluation score if available.
        4. Merges memory labels from context.
        5. Sets the system prompt for adversarial chat.

        Args:
            context (MultiTurnAttackContext): Attack context with configuration

        Raises:
            ValueError: If the system prompt is not defined.
        """
        # Initialize the conversation session if not already set
        context.session = context.session or ConversationSession()
        logger.debug(f"Conversation session ID: {context.session.conversation_id}")
        logger.debug(f"Adversarial chat conversation ID: {context.session.adversarial_chat_conversation_id}")

        # Initialize the conversation state
        conversation_state: ConversationState = self._conversation_manager.initialize_conversation_with_history(
            target=context.target,
            max_turns=context.max_turns,
            conversation_id=context.session.conversation_id,
            history=self._prepend_conversation,
        )

        # update the turns based on prepend conversation
        context.executed_turns = conversation_state.turn_count

        # update the custom prompt if provided
        if RedTeamingStrategy._has_custom_prompt(state=conversation_state):
            context.custom_prompt = conversation_state.last_user_message

        # get the last assistant message evaluation score if available
        # and add it to the score evaluator 
        score = self._retrieve_last_assistant_message_evaluation_score(
            state=conversation_state
        )
        self._score_evaluator.add(score=score)

        # update the memory labels 
        context.memory_labels = combine_dict(existing_dict=self._memory_labels, new_dict=context.memory_labels)

        # set the system prompt for the adversarial chat
        if self._system_prompt is None:
            raise ValueError("System prompt must be set")
        
        context.adversarial_chat.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=context.session.adversarial_chat_conversation_id,
            orchestrator_identifier=self.identifier,
            labels=context.memory_labels,
        )
        
    async def execute(self, *, context: MultiTurnAttackContext) -> MultiTurnAttackResult:
        """
        Execute the red teaming attack by iteratively generating prompts,
        sending them to the target, and scoring the responses in a loop
        until the objective is achieved or the maximum turns are reached.

        Algorithm:
        1. Set up the conversation state and log relevant configurations.
        2. For each turn, generate a new adversarial prompt and send it to the target.
        3. Score the target's response to gauge progress and determine if the objective is met.
        4. If the objective is not met and backtracking is enabled, apply backtracking steps.
            Note: this attack currently uses no backtracking strategy.
        5. Record results and finalize the attack execution once the loop ends.

        Args:
            context: Attack context with configuration, including objective, memory, and constraints.

        Returns:
            A MultiTurnAttackResult containing details about the execution, such as the final response,
            last score, achieved objective status, and number of turns executed.
        """
        if not context.objective:
            raise ValueError("Attack objective must be set")
            
        start_time = time.time()
        
        # Set up the initial conversation state
        logger.info(f"Starting red teaming attack with objective: {context.objective}")
        logger.info(f"Max turns: {context.max_turns}")
        logger.info(f"Backtracking strategy: {context.backtracking_strategy.value if context.backtracking_strategy else 'None'}")
        
        # Execute conversation turns
        # Attack Flow:
        # 1. Generate adversarial prompt based on previous feedback or custom prompt.
        # 2. Send the generated prompt to the target system.
        # 3. Evaluate the target's response using the objective scorer.
        # 4. Check if the attack objective has been achieved.
        # 5. If objective is not achieved and backtracking is enabled, apply backtracking.
        # 6. Repeat steps 1-5 until objective is achieved or max turns are reached.
        while context.executed_turns < context.max_turns and not context.achieved_objective:
            # Determine what to send next
            prompt_to_send = await self._generate_next_prompt(context=context)
            
            # Send message to target
            target_response = await self._send_to_target(context, prompt_to_send)
            context.last_response = target_response
            context.executed_turns += 1
            
            # Score the response
            score = await self._score_response(context=context, response=target_response)
            context.last_score = score
            
            # We probably can record turn? I just leave it here for now
            # as a tuple of prompt, response, turn number, and score
            # then the state could be tracked easily internally 
            # TODO: implement this in the future
            
            # Check if objective achieved
            context.achieved_objective = await self._evaluate_objective_achievement(context, target_response, score)
                
            # Apply backtracking if configured and needed
            if (not context.achieved_objective and 
                context.backtracking_strategy and 
                context.executed_turns < context.max_turns):
                await self._apply_backtracking(context)
        
        # Log the result of the attack
        if context.achieved_objective:
            logger.info(f"Objective achieved on turn {context.executed_turns}")
        else:
            logger.info(
                "The red teaming orchestrator has not achieved the objective after the maximum "
                f"number of turns ({context.max_turns}).",
            )

        # Prepare the result
        execution_time = int((time.time() - start_time) * 1000)
        logger.info(f"Red teaming attack completed in {execution_time} ms")

        return MultiTurnAttackResult(
            conversation_id=context.session.conversation_id,
            objective=context.objective,
            achieved_objective=context.achieved_objective,
            executed_turns=context.executed_turns,
            last_response=context.last_response,
            last_score=context.last_score,
            memory_labels=context.memory_labels
        )
    
    async def _generate_next_prompt(self, context: MultiTurnAttackContext) -> str:
        """
        Generate the next prompt to be sent to the target during the red teaming attack.

        This method is called each turn to obtain fresh adversarial text based on previous feedback,
        error states, or the custom prompt if it's the first turn. It integrates feedback from the
        scorer when available, and handles blocked or error responses by returning fallback prompts.

        Args:
            context: The attack context containing state such as custom prompts, turn counts, and past responses.

        Returns:
            A string representing the next adversarial prompt to send to the target.
        """
        # If first turn and custom prompt provided, use it
        if context.executed_turns == 0 and context.custom_prompt:
            logger.debug("Using custom prompt for first turn")
            return context.custom_prompt
            
        # Generate prompt using adversarial chat
        logger.debug(f"Generating prompt for turn {context.executed_turns + 1}")
        
        # Prepare prompt for the adversarial chat
        prompt_text = await self._build_adversarial_prompt(context)
        
        # send the prompt to adversarial chat
        prompt_grp = SeedPromptGroup(prompts=[
            SeedPrompt(value=prompt_text, data_type="text")
        ])

        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_grp,
            conversation_id=context.session.adversarial_chat_conversation_id,
            target=context.adversarial_chat,
            orchestrator_identifier=self.identifier,
            labels=context.memory_labels,
        )
        return response.get_value()
    
    async def _build_adversarial_prompt(
        self, 
        context: MultiTurnAttackContext, 
    ) -> str:
        """Build a prompt for the adversarial chat.
        
        Args:
            context: Attack context
            conversation_turns: Previous conversation turns
            
        Returns:
            Formatted prompt for adversarial chat
        """
        response = self._conversation_manager.get_last_message(
            conversation_id=context.session.adversarial_chat_conversation_id, 
            role="assistant"
        )
        if not response:
            return self._chat_seed_prompt.value
        
        if response.converted_value_data_type in ["text", "error"]:
            return self._handle_adversarial_text_response(response=response)
        return self._handle_adversarial_file_response(response=response)
    
    def _handle_adversarial_text_response(self, *, response: PromptRequestPiece) -> str:
        """
        Handle the text response from the adversarial chat by appending any
        available scoring feedback to the returned text. If the response 
        indicates a block or error, return a fallback message instead.

        Args:
            response (PromptRequestPiece): The response from the adversarial chat

        Returns:
            str: The text to be sent to the adversarial chat in the next turn
        """
        feedback = self._score_evaluator.get_feedback()
        if not response.has_error():
            # if response has no error, we can use the converted value
            prompt_text = response.converted_value

            # if we have feedback, append it to the prompt
            # to provide more context to the adversarial chat
            if feedback:
                prompt_text += f"\n\n{feedback}"
            return prompt_text
        
        elif response.is_blocked():
            return DEFAULT_ADVERSARIAL_PROMPT_IF_OBJECTIVE_TARGET_IS_BLOCKED
        
        return f"Request to target failed: {response.response_error}"
    
    def _handle_adversarial_file_response(self, *, response: PromptRequestPiece) -> str:
        """
        Handle the file response from the adversarial chat.

        If the response indicates an error, raise a RuntimeError. When scoring is disabled (use_score_as_feedback=False)
        or no scoring rationale is provided, raise a ValueError. Otherwise, return the textual feedback as the prompt.

        Args:
            response (PromptRequestPiece): The response containing file or non-text data.

        Returns:
            str: The suitable feedback or error message to pass back to the adversarial chat.
        """
        if response.has_error():
            raise RuntimeError(
                "Request to target failed despite the returned data type "
                f"{response.converted_value_data_type}: "
                f"{response.response_error}"
            )
        
        if not self._use_score_as_feedback:
            raise ValueError(
                f"{DEFAULT_ERR_MSG_IF_OBJECTIVE_TARGET_HAS_NON_TEXT_RESPONSE}"
                "However, the use_score_as_feedback flag is set to False so it cannot be utilized."
            )
        
        feedback = self._score_evaluator.get_feedback()
        if not feedback:
            raise ValueError(
                f"{DEFAULT_ERR_MSG_IF_OBJECTIVE_TARGET_HAS_NON_TEXT_RESPONSE}" 
                "However, no scoring rationale was provided by the scorer."
            )
        
        return feedback
    
    async def _send_to_target(self, context: MultiTurnAttackContext, prompt: str) -> PromptRequestPiece:
        """
        Send a prompt to the target system.

        Constructs a seed prompt group, sends it to the target via the prompt normalizer, 
        and returns the response as a PromptRequestPiece.

        Args:
            context (MultiTurnAttackContext): The current attack context.
            prompt (str): The prompt to send to the target.

        Returns:
            PromptRequestPiece: The system's response to the prompt.
        """

        converter_cfgs = [PromptConverterConfiguration(converters=context.prompt_converters)]

        logger.info(f"Sending prompt to target: {prompt[:50]}...")
        
        # Create a seed prompt group from the prompt
        seed_prompt = SeedPrompt(value=prompt, data_type="text")
        seed_prompt_group = SeedPromptGroup(prompts=[seed_prompt])
            
        # Send the prompt to the target
        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            conversation_id=context.session.conversation_id,
            request_converter_configurations=converter_cfgs,
            target=context.target,
            labels=context.memory_labels,
            orchestrator_identifier=self.identifier,
        )

        return response.get_piece()
    
    async def _score_response(self, *, context: MultiTurnAttackContext, response: PromptRequestPiece) -> Optional[Score]:
        """
        Evaluate the target's response with the objective scorer.

        Checks if the response is blocked or has an error before scoring. 
        Returns the resulting Score object or None if the response was blocked.

        Args:
            response (PromptRequestPiece): The target system's response.

        Returns:
            Optional[Score]: The score of the response if available, otherwise None.
        """
        if not response.has_error():
            return await self._score_evaluator.evaluate_response(
                response=response,
                only_allowed_type="true_false",
            )
        elif response.is_blocked():
            return None
        
        raise RuntimeError(f"Response error: {response.response_error}")
    
    async def _evaluate_objective_achievement(
        self, 
        context: MultiTurnAttackContext, 
        response: PromptRequestPiece, 
        score: Optional[Score]
    ) -> bool:
        """
        Evaluate whether the objective has been achieved using a basic check.

        NOTE: Although we currently only call is_objective_achieved on the score,
        this function is kept as an example for future expansions where more
        complex evaluation logic could be implemented.

        Args:
            context (MultiTurnAttackContext): The current attack context.
            response (PromptRequestPiece): The response from the target.
            score (Optional[Score]): The score for the response.

        Returns:
            bool: True if the objective has been achieved, False otherwise.
        """
        return self._score_evaluator.is_objective_achieved(
            score=score
        )
    
    async def _apply_backtracking(self, context: MultiTurnAttackContext) -> None:
        """
        Apply backtracking to the conversation if needed.

        NOTE: Even though red teaming does not currently use any backtracking strategy,
        this is kept as an example for future attack strategies.

        Args:
            context (MultiTurnAttackContext): The attack context.
        """
        # This should be implemented with proper backtracking logic
        # For simplicity, we'll just log the backtracking event
        logger.debug(f"Applying backtracking strategy: {context.backtracking_strategy.value}")
        pass
    
    @staticmethod
    def _has_custom_prompt(state: ConversationState) -> bool:
        """
        Check if the last user message is considered a custom prompt.

        A custom prompt is assumed if the user message exists and no assistant
        message scores are present, indicating a fresh prompt not yet evaluated.

        Args:
            state (ConversationState): The conversation state.

        Returns:
            bool: True if the last user message is a custom prompt; otherwise, False.
        """
        return state.last_user_message and not state.last_assistant_message_scores
    
    def _retrieve_last_assistant_message_evaluation_score(
        self,
        state: ConversationState
    ) -> Optional[Score]:
        """
        Retrieve the last assistant message evaluation score.

        Args:
            state (ConversationState): The conversation state

        Returns:
            Optional[Score]: The score of the last assistant message
        """
        if not state.last_assistant_message_scores:
            return None
        
        objective_score: Optional[Score] = None
        for score in state.last_assistant_message_scores:
            if score.scorer_class_identifier["__type__"] == self._score_evaluator.scorer_type:
                objective_score = score
                break
        return objective_score