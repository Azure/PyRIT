# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import concurrent.futures
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional, overload

from aioconsole import ainput

from pyrit.common.utils import combine_dict, get_kwarg_param
from pyrit.executor.core import StrategyConverterConfig
from pyrit.executor.workflow.core import (
    WorkflowContext,
    WorkflowResult,
    WorkflowStrategy,
)
from pyrit.models import Score, SeedPrompt, SeedPromptGroup
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@dataclass
class XPIAContext(WorkflowContext):
    """
    Context for Cross-Domain Prompt Injection Attack (XPIA) workflow.

    Contains execution-specific parameters needed for each XPIA attack run.
    Immutable objects like targets and scorers are stored in the workflow instance.
    """

    # The content to attack the processing target with (e.g., a jailbreak)
    attack_content: str

    # Callback to execute after the attack prompt is positioned in the attack location
    processing_callback: Callable[[], Awaitable[str]]

    # Conversation ID for the attack setup target
    attack_setup_target_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Conversation ID for the processing phase
    processing_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # The prompt to send to the processing target
    processing_prompt: str = ""

    # Additional labels that can be applied throughout the workflow
    memory_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class XPIAResult(WorkflowResult):
    """
    Result of XPIA workflow execution.

    Contains the outcome of the cross-domain prompt injection attack, including
    the processing response, optional score, and attack setup response.
    """

    # Response from the processing target
    processing_response: str

    # Score if a scorer was used, None otherwise
    score: Optional[Score] = None

    # Response from the attack setup target
    attack_setup_response: Optional[str] = None

    @property
    def success(self) -> bool:
        """
        Determine if the attack was successful based on the score.

        Returns:
            bool: True if the attack was successful (score exists and has a positive value),
                False otherwise.
        """
        if self.score is None:
            return False
        score_value = self.score.get_value()
        return bool(score_value) if isinstance(score_value, (int, float)) else False


class XPIAWorkflow(WorkflowStrategy[XPIAContext, XPIAResult]):
    """
    Implementation of Cross-Domain Prompt Injection Attack (XPIA) workflow.

    This workflow orchestrates an attack where:
    1. An attack prompt is generated and positioned using the attack_setup_target
    2. The processing_callback is executed to trigger the target's processing
    3. The response is optionally scored to determine success

    The workflow supports customization through prompt converters and scorers,
    allowing for various attack techniques and evaluation methods.
    """

    def __init__(
        self,
        *,
        attack_setup_target: PromptTarget,
        scorer: Optional[Scorer] = None,
        converter_config: Optional[StrategyConverterConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        logger: logging.Logger = logger,
    ):
        """
        Initialize the XPIA workflow.

        Args:
            attack_setup_target (PromptTarget): The target that generates the attack prompt
                and gets it into the attack location.
            scorer (Optional[Scorer]): Optional scorer to evaluate the processing response.
                If no scorer is provided the workflow will skip scoring.
            converter_config (Optional[StrategyConverterConfig]): Optional converter
                configuration for request and response converters.
            prompt_normalizer (Optional[PromptNormalizer]): Optional PromptNormalizer
                instance. If not provided, a new one will be created.
            logger (logging.Logger): Logger instance for logging events.
        """
        super().__init__(context_type=XPIAContext, logger=logger)
        self._attack_setup_target = attack_setup_target
        self._scorer = scorer

        converter_config = converter_config or StrategyConverterConfig()
        self._request_converters = converter_config.request_converters
        self._response_converters = converter_config.response_converters

        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

    def _validate_context(self, *, context: XPIAContext) -> None:
        """
        Validate the XPIA context before execution.

        This method ensures that all required parameters are present and valid
        before proceeding with the attack workflow execution.

        Args:
            context (XPIAContext): The context to validate.

        Raises:
            ValueError: If the context is invalid (missing attack_content or processing_callback).
        """
        if not context.attack_content:
            raise ValueError("attack_content cannot be empty")

        if not context.processing_callback:
            raise ValueError("processing_callback is required")

    async def _setup_async(self, *, context: XPIAContext) -> None:
        """
        Setup phase before executing the workflow.

        This method prepares the execution context by generating conversation IDs
        and combining memory labels for the workflow execution.

        Args:
            context (XPIAContext): The context for the workflow. This will be modified
                to include setup-specific configuration.
        """
        context.attack_setup_target_conversation_id = str(uuid.uuid4())
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

    async def _perform_async(self, *, context: XPIAContext) -> XPIAResult:
        """
        Execute the XPIA workflow.

        This method orchestrates the complete XPIA attack by:
        1. Sending the attack content to the attack setup target
        2. Executing the processing callback to trigger target processing
        3. Optionally scoring the processing response to determine success

        Args:
            context (XPIAContext): The context containing workflow parameters including
                attack content, processing callback, and memory labels.

        Returns:
            XPIAResult: The result of the workflow execution containing the processing
                response, optional score, and attack setup response.
        """
        # Step 1: Setup and send attack prompt
        setup_response_text = await self._setup_attack_async(context=context)

        # Step 2: Execute processing callback
        processing_response = await self._execute_processing_async(context=context)

        # Step 3: Score the response if scorer is provided
        score = await self._score_response_async(processing_response=processing_response)

        return XPIAResult(
            processing_response=processing_response,
            score=score,
            attack_setup_response=setup_response_text,
        )

    async def _setup_attack_async(self, *, context: XPIAContext) -> str:
        """
        Setup and send the attack prompt to the attack setup target.

        This method creates a seed prompt group from the attack content and sends it
        to the attack setup target using configured request converters.

        Args:
            context (XPIAContext): The context containing the attack content and labels.

        Returns:
            str: The response text from the attack setup target.
        """
        self._logger.info(
            "Sending the following prompt to the prompt target (after applying prompt "
            f'converter operations) "{context.attack_content}"',
        )

        # Create seed prompt group from attack content
        seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=context.attack_content, data_type="text")])

        setup_response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters,
            target=self._attack_setup_target,
            labels=context.memory_labels,
            orchestrator_identifier=self.get_identifier(),
            conversation_id=context.attack_setup_target_conversation_id,
        )

        setup_response_text = setup_response.get_value()
        self._logger.info(f'Received the following response from the prompt target: "{setup_response_text}"')

        return setup_response_text

    async def _execute_processing_async(self, *, context: XPIAContext) -> str:
        """
        Execute the processing callback to trigger target processing.

        This method calls the processing callback function to execute the processing
        target and retrieve the response.

        Args:
            context (XPIAContext): The context containing the processing callback.

        Returns:
            str: The response from the processing target.
        """
        processing_response = await context.processing_callback()
        self._logger.info(f'Received the following response from the processing target "{processing_response}"')
        return processing_response

    async def _score_response_async(self, *, processing_response: str) -> Optional[Score]:
        """
        Score the processing response if a scorer is provided.

        This method uses the configured scorer to evaluate the processing response
        and determine the success of the attack. The scoring is executed using
        asyncio to match the legacy implementation behavior.

        Args:
            processing_response (str): The response from the processing target to score.

        Returns:
            Optional[Score]: The score if a scorer is configured, None otherwise.
        """
        if not self._scorer:
            self._logger.info("No scorer provided. Returning raw processing response.")
            return None

        # Use thread pool executor to match legacy behavior
        pool = concurrent.futures.ThreadPoolExecutor()
        try:
            score = pool.submit(asyncio.run, self._scorer.score_text_async(processing_response)).result()[0]
            self._logger.info(f"Score of the processing response: {score}")
            return score
        except Exception as e:
            self._logger.error(f"Error scoring response: {e}")
            return None
        finally:
            pool.shutdown(wait=False)

    async def _teardown_async(self, *, context: XPIAContext) -> None:
        """
        Teardown phase after executing the workflow.

        This method performs cleanup operations after the XPIA workflow execution.
        Currently, no specific teardown operations are required for the base workflow,
        but this method can be overridden by subclasses if needed.

        Args:
            context (XPIAContext): The context for the workflow.
        """
        # No specific teardown operations required for base XPIA workflow
        pass

    @overload
    async def execute_async(
        self,
        *,
        attack_content: str,
        processing_prompt: Optional[str] = None,
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> XPIAResult:
        """
        Execute the XPIA workflow strategy asynchronously with the provided parameters.

        Args:
            attack_content (str): The content to use for the attack.
            processing_prompt (Optional[str]): The prompt to send to the processing target. This should include
                placeholders to invoke plugins (if any).
            memory_labels (Optional[Dict[str, str]]): Memory labels for the attack context.
            **kwargs: Additional parameters for the attack.

        Returns:
            XPIAResult: The result of the workflow execution.
        """
        ...

    @overload
    async def execute_async(
        self,
        **kwargs,
    ) -> XPIAResult: ...

    async def execute_async(
        self,
        **kwargs,
    ) -> XPIAResult:
        """
        Execute the XPIA workflow strategy asynchronously with the provided parameters.
        """
        attack_content = get_kwarg_param(kwargs=kwargs, param_name="attack_content", expected_type=str)
        processing_prompt = get_kwarg_param(
            kwargs=kwargs, param_name="processing_prompt", expected_type=str, required=False
        )
        memory_labels = get_kwarg_param(kwargs=kwargs, param_name="memory_labels", expected_type=dict, required=False)

        return await super().execute_async(
            attack_content=attack_content,
            processing_prompt=processing_prompt or "",
            memory_labels=memory_labels or {},
            **kwargs,
        )


class XPIATestWorkflow(XPIAWorkflow):
    """
    XPIA workflow with automated test processing.

    This variant automatically handles the processing phase by sending
    a predefined prompt to a processing target. It is designed for automated
    testing scenarios where the processing can be scripted rather than manual.

    The workflow creates an automated processing callback that sends the
    processing prompt to the configured processing target and returns the response.
    """

    def __init__(
        self,
        *,
        attack_setup_target: PromptTarget,
        processing_target: PromptTarget,
        scorer: Scorer,
        converter_config: Optional[StrategyConverterConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        logger: logging.Logger = logger,
    ) -> None:
        """
        Initialize the XPIA test workflow.

        Args:
            attack_setup_target (PromptTarget): The target that generates the attack prompt
                and gets it into the attack location.
            processing_target (PromptTarget): The target of the attack which processes the
                processing prompt. This should include references to invoke plugins (if any).
            scorer (Scorer): The scorer to use to score the processing response. This is
                required for test workflows to evaluate attack success.
            converter_config (Optional[StrategyConverterConfig]): Optional converter
                configuration for request and response converters.
            prompt_normalizer (Optional[PromptNormalizer]): Optional PromptNormalizer
                instance. If not provided, a new one will be created.
            logger (logging.Logger): Logger instance for logging events.
        """
        super().__init__(
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            converter_config=converter_config,
            prompt_normalizer=prompt_normalizer,
            logger=logger,
        )
        self._processing_target = processing_target

    def _validate_context(self, *, context: XPIAContext) -> None:
        """
        Validate the XPIA test context.

        This method validates the context for test workflow execution, ensuring
        that both attack content and processing prompt are provided.

        Args:
            context (XPIAContext): The context to validate.

        Raises:
            ValueError: If the context is invalid (missing attack_content or processing_prompt).
        """
        super()._validate_context(context=context)

        if not context.processing_prompt:
            raise ValueError("processing_prompt cannot be empty")

    async def _setup_async(self, *, context: XPIAContext) -> None:
        """
        Setup phase for XPIA test workflow execution.

        This method creates an automated processing callback that sends the processing
        prompt to the configured processing target. The callback is attached to the
        context for use during workflow execution.

        Args:
            context (XPIAContext): The execution context containing the processing prompt
                and configuration settings. This context will be modified to include
                the processing callback.
        """

        # Create the processing callback using the test context
        async def process_async() -> str:
            seed_prompt = SeedPrompt(value=context.processing_prompt, data_type="text")
            seed_prompt_group = SeedPromptGroup(prompts=[seed_prompt])

            response = await self._prompt_normalizer.send_prompt_async(
                seed_prompt_group=seed_prompt_group,
                target=self._processing_target,
                request_converter_configurations=self._request_converters,
                response_converter_configurations=self._response_converters,
                labels=context.memory_labels,
                orchestrator_identifier=self.get_identifier(),
                conversation_id=context.processing_conversation_id,
            )

            return response.get_value()

        # Set the processing callback on the context
        context.processing_callback = process_async
        return await super()._setup_async(context=context)


class XPIAManualProcessingWorkflow(XPIAWorkflow):
    """
    XPIA workflow with manual processing intervention.

    This variant pauses execution to allow manual triggering of the
    processing target, then accepts the output via console input.
    This is useful for scenarios where the processing target requires
    manual interaction or cannot be automated.

    The workflow will prompt the operator to manually trigger the processing
    target's execution and paste the output into the console for scoring.
    """

    def __init__(
        self,
        *,
        attack_setup_target: PromptTarget,
        scorer: Scorer,
        converter_config: Optional[StrategyConverterConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        logger: logging.Logger = logger,
    ) -> None:
        """
        Initialize the XPIA manual processing workflow.

        Args:
            attack_setup_target (PromptTarget): The target that generates the attack prompt
                and gets it into the attack location.
            scorer (Scorer): The scorer to use to score the processing response. This is
                required to evaluate the manually provided response.
            converter_config (Optional[StrategyConverterConfig]): Optional converter
                configuration for request and response converters.
            prompt_normalizer (Optional[PromptNormalizer]): Optional PromptNormalizer
                instance. If not provided, a new one will be created.
            logger (logging.Logger): Logger instance for logging events.
        """
        super().__init__(
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            converter_config=converter_config,
            prompt_normalizer=prompt_normalizer,
            logger=logger,
        )

    def _validate_context(self, *, context: XPIAContext) -> None:
        """
        Validate the XPIA manual processing context.

        This method validates the context for manual processing workflow execution.
        Since the processing callback will be created automatically, we only need
        to validate that the attack content is present.

        Args:
            context (XPIAContext): The context to validate.

        Raises:
            ValueError: If the context is invalid (missing attack_content).
        """
        # Skip the base validation for processing_callback since we'll set it ourselves
        if not context.attack_content:
            raise ValueError("attack_content cannot be empty")

    async def _setup_async(self, *, context: XPIAContext) -> None:
        """
        Setup phase for XPIA manual processing workflow execution.

        This method creates a manual input callback that prompts the operator
        to trigger the processing target's execution and paste the output.

        Args:
            context (XPIAContext): The execution context. This context will be
                modified to include the manual processing callback.
        """

        # Create the manual input callback
        async def manual_input_async() -> str:
            return await ainput("Please trigger the processing target's execution and paste the output here: ")

        # Set the processing callback on the context
        context.processing_callback = manual_input_async
        return await super()._setup_async(context=context)
