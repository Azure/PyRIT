from __future__ import annotations
import logging
from typing import List, Optional

from pyrit.common.utils import combine_dict
from pyrit.models import PromptDataType, PromptRequestResponse
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.orchestratorv3.components.conversation_manager import ConversationManager, ConversationState
from pyrit.orchestratorv3.components.prompt_utils import PromptUtils
from pyrit.orchestratorv3.models.builder import AttackStrategy
from pyrit.orchestratorv3.models.context import SingleTurnAttackContext, SingleTurnAttackResult
from pyrit.orchestratorv3.single_turn.base import SingleTurnBaseAttackBuilder, SingleTurnBaseOrchestratorBuilder
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)

class PromptSendingOrchestratorBuilder(SingleTurnBaseOrchestratorBuilder):
    """Builder for configuring prompt sending orchestrators."""
    
    def __init__(
        self, 
        *,
        target: PromptTarget, 
        batch_size: int = 10, 
        prompt_normalizer: Optional[PromptNormalizer] = None,
        prompt_converters: Optional[List[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None
    ):
        """
        Initialize the PromptSending orchestrator builder.
        
        Args:
            target: Target system to attack
            batch_size: Batch size for sending prompts
            prompt_normalizer: Optional custom prompt normalizer
            prompt_converters: Optional list of prompt converters
            memory: Optional memory interface
        """
        super().__init__(
            target=target,
            batch_size=batch_size,
            prompt_normalizer=prompt_normalizer,
            memory=memory
        )
        if prompt_converters:
            self.with_prompt_converters(prompt_converters)
    
    def attack(self) -> PromptSendingAttackBuilder:
        """Transition to attack configuration.
        
        Returns:
            A prompt sending attack builder
        """
        return PromptSendingAttackBuilder(
            context=self.context,
            prompt_normalizer=self._prompt_normalizer,
            memory=self._memory
        )

class PromptSendingAttackBuilder(SingleTurnBaseAttackBuilder):
    """
    Builder for configuring and executing prompt sending attacks.

    This class extends SingleTurnBaseAttackBuilder to provide functionality for:
        1. Appending conversation history before request creation if needed.
        2. Preparing NormalizerRequest objects for each prompt.
        3. Executing the final attack using PromptSendingAttackStrategy.

    It centralizes the setup and orchestration of all components needed to send
    prompts to a target system in a single-turn attack scenario.
    """
    
    def __init__(
        self, 
        *,
        context: SingleTurnAttackContext,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        memory: Optional[MemoryInterface] = None
    ):
        """Initialize the Prompt Sending Attack Builder.
        
        Args:
            context: Context for the attack
            prompt_normalizer: Optional custom prompt normalizer
            memory: Optional memory interface
        """
        super().__init__(
            context=context,
            prompt_normalizer=prompt_normalizer,
            memory=memory
        )
        self._prepended_conversation = []
    
    async def execute(self) -> SingleTurnAttackResult:
        """Execute the attack with the current configuration.
        
        Returns:
            Result of the attack
        """
        if not self._context.prompts or len(self._context.prompts) == 0:
            raise ValueError("No prompts configured for attack")
        
        ctx = self.context
            
        strategy = PromptSendingAttackStrategy(
            prompt_normalizer=self._prompt_normalizer,
            memory=self._memory,
            prepended_conversation=self._prepended_conversation,
            scorers=self._scorers
        )
        
        await strategy.setup(context=ctx)
        return await strategy.execute(context=ctx)

class PromptSendingAttackStrategy(AttackStrategy[SingleTurnAttackContext, SingleTurnAttackResult]):
    """
    Strategy for sending prompts to a target.

    This strategy handles preparing, sending, and optionally scoring prompt responses.
    """
    
    def __init__(
        self,
        *,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        prepended_conversation: Optional[List[PromptRequestResponse]] = None,
        memory: Optional[MemoryInterface] = None,
        scorers: Optional[List[Scorer]] = None
    ):
        """
        Initialize the strategy.
        
        Args:
            prompt_normalizer: Optional custom prompt normalizer
            prepended_conversation: Optional conversation to prepend
            memory: Optional memory interface
            scorers: Optional list of scorers
        """
        super().__init__(
            memory=memory,
            prompt_normalizer=prompt_normalizer
        )
        self._prepended_conversation = prepended_conversation
        self._scorers = scorers or []
        self._conversation_manager = ConversationManager(
            orchestrator_identifier=self.identifier,
            memory=memory,
        )
        
    async def setup(self, *, context: SingleTurnAttackContext) -> None:
        """
        Prepare the strategy for the single-turn attack. This method is called before execution 
        to make sure memory labels are up to date and any additional setup tasks are performed.

        Args:
            context (SingleTurnAttackContext): The context for the single-turn attack, 
            containing target information, memory labels, batch size, and other relevant data.
        """
        # Get updated memory labels
        context.memory_labels = combine_dict(existing_dict=self._memory_labels, new_dict=context.memory_labels)
            
    async def execute(self, *, context: SingleTurnAttackContext) -> SingleTurnAttackResult:
        """
        Executes a single-turn attack by generating prompt requests, sending them to the target, 
        and optionally scoring the responses.

        Args:
            context (SingleTurnAttackContext): The context for the single-turn attack, 
            containing target information, memory labels, batch size, and other relevant data.

        Returns:
            SingleTurnAttackResult: The result of the single-turn attack, including the list of 
            responses received from the target.
        """
        
        # Generate prompt requests based on the context
        # 1) Build normalized prompt requests
        # 2) Send requests to target in batches
        # 3) Score and return the response results
        requests = await self._get_prompt_requests(
            context=context,
        )
                
        # Send prompts
        responses = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=requests,
            target=context.target,
            labels=context.memory_labels,
            orchestrator_identifier=self.identifier,
            batch_size=context.batch_size,
        )
        
        # Score responses if scorers were provided
        await self._score_responses(
            responses=responses,
            context=context
        )
                
        # Create and return the result
        return SingleTurnAttackResult(
            orchestrator_id=self._identifier.id,
            prompt_list=responses
        )
    
    async def _get_prompt_requests(
        self, 
        *,
        context: SingleTurnAttackContext,
        prompt_type: PromptDataType = "text",
    ) -> List[NormalizerRequest]:
        """
        Create prompt requests for the given context.
        Args:
            context: The context for the attack
            prompt_type: The type of prompt to create
        Returns:
            List of NormalizerRequest objects
        """
        if not context.prompts:
            raise ValueError("No prompts configured for attack")

        requests = PromptUtils.build_normalizer_requests(
            prompts=context.prompts,
            prompt_type=prompt_type,
            converters=context.prompt_converters,
            metadata=context.metadata,
        )

        # Handle prepended conversation if needed
        if self._prepended_conversation:
            for request in requests:
                await self._prepare_conversation(request)

        return requests
    
    async def _score_responses(
        self, 
        *,
        responses: List[PromptRequestResponse],
        context: SingleTurnAttackContext
    ) -> None:
        """
        Score the responses using the provided scorers.
        
        Args:
            responses: The responses to score
            context: The context for the attack
        """
        if not self._scorers or not responses:
            return
        
        response_pieces = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)
        
        for scorer in self._scorers:
            await scorer.score_responses_inferring_tasks_batch_async(
                request_responses=response_pieces, 
                batch_size=context.batch_size
            )

    async def _prepare_conversation(self, *, normalizer_request: NormalizerRequest) -> None:
        """
        Prepares the conversation by initializing it with the prepended conversation.
        
        Args:
            normalizer_request: The normalizer request to prepare
        """
        if not self._prepended_conversation:
            return
        
        # Initialize conversation with prepended conversation
        conversation_id = normalizer_request.conversation_id
        self._conversation_manager.initialize_conversation_with_history(
            conversation_id=conversation_id,
            history=self._prepended_conversation,
        )