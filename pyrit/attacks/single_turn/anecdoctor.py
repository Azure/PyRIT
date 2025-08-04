# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, overload

import yaml

from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.base.attack_context import AttackContext
from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.common.path import DATASETS_PATH
from pyrit.common.utils import get_kwarg_param
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@dataclass
class AnecdoctorAttackContext(AttackContext):
    """
    Context specific to Anecdoctor attacks.

    Contains all parameters needed for executing an Anecdoctor attack,
    including the evaluation data, language settings, and conversation ID.
    """

    # The data in ClaimsReview format to use in constructing the prompt
    evaluation_data: List[str] = field(default_factory=list)

    # The language of the content to generate
    language: str = "english"

    # The type of content to generate (e.g., "viral tweet", "news article")
    content_type: str = "viral tweet"

    # Conversation ID for the main attack (generated if not provided)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def create_from_params(
        cls,
        *,
        objective: str,
        prepended_conversation: List[PromptRequestResponse],
        memory_labels: Dict[str, str],
        **kwargs,
    ) -> AnecdoctorAttackContext:
        """
        Create AnecdoctorAttackContext from parameters.

        Args:
            objective (str): The natural-language description of the attack's objective.
            prepended_conversation (List[PromptRequestResponse]): Conversation to prepend to the target model
            memory_labels (Dict[str, str]): Labels for memory management.
            **kwargs: Additional keyword arguments.

        Returns:
            AnecdoctorAttackContext: The created context instance.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        # Extract and validate evaluation_data (required)
        content_type = (
            get_kwarg_param(
                kwargs=kwargs,
                param_name="content_type",
                expected_type=str,
            )
            or ""
        )

        language = (
            get_kwarg_param(
                kwargs=kwargs,
                param_name="language",
                expected_type=str,
            )
            or ""
        )

        evaluation_data = (
            get_kwarg_param(
                kwargs=kwargs,
                param_name="evaluation_data",
                expected_type=list,
            )
            or []
        )

        # Create instance with all parameters
        return cls(
            objective=objective,
            memory_labels=memory_labels,
            evaluation_data=evaluation_data,
            language=language,
            content_type=content_type,
        )


class AnecdoctorAttack(AttackStrategy[AnecdoctorAttackContext, AttackResult]):
    """
    Implementation of the Anecdoctor attack strategy.

    The Anecdoctor attack generates misinformation content by either:
    1. Using few-shot examples directly (default mode when processing_model is not provided)
    2. First extracting a knowledge graph from examples, then using it (when processing_model is provided)

    This attack is designed to test a model's susceptibility to generating false or
    misleading content when provided with examples in ClaimsReview format. The attack
    can optionally use a processing model to extract a knowledge graph representation
    of the examples before generating the final content.

    The attack flow consists of:
    1. (Optional) Extract knowledge graph from evaluation data using processing model
    2. Format a system prompt based on language and content type
    3. Send formatted examples (or knowledge graph) to target model
    4. (Optional) Score the response if scoring is configured
    5. Return the generated content as the attack result
    """

    # Class-level constants for YAML file paths
    _ANECDOCTOR_BUILD_KG_YAML = "anecdoctor_build_knowledge_graph.yaml"
    _ANECDOCTOR_USE_KG_YAML = "anecdoctor_use_knowledge_graph.yaml"
    _ANECDOCTOR_USE_FEWSHOT_YAML = "anecdoctor_use_fewshot.yaml"
    _ANECDOCTOR_PROMPT_PATH = Path("orchestrators", "anecdoctor")

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        processing_model: Optional[PromptChatTarget] = None,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
    ) -> None:
        """
        Initialize the Anecdoctor attack strategy.

        Args:
            objective_target (PromptChatTarget): The chat model to be evaluated/attacked.
            processing_model (Optional[PromptChatTarget]): The model used for knowledge graph extraction.
                If provided, the attack will extract a knowledge graph from the examples before generation.
                If None, the attack will use few-shot examples directly.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.
            prompt_normalizer (Optional[PromptNormalizer]): Normalizer for handling prompts.
        """
        # Initialize base class
        super().__init__(logger=logger, context_type=AnecdoctorAttackContext)

        # Store configuration
        self._objective_target = objective_target
        self._processing_model = processing_model
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

        # Initialize converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()

        # Check for unused optional parameters and warn if they are set
        self._warn_if_set(config=attack_scoring_config, unused_fields=["refusal_scorer"])

        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._objective_scorer = attack_scoring_config.objective_scorer
        if self._objective_scorer and self._objective_scorer.scorer_type != "true_false":
            raise ValueError("Objective scorer must be a true/false scorer")

        self._kg_prompt_template = self._load_prompt_from_yaml(yaml_filename=self._ANECDOCTOR_BUILD_KG_YAML)
        # Prepare the system prompt based on whether we're using knowledge graph
        if self._processing_model:
            self._system_prompt_template = self._load_prompt_from_yaml(yaml_filename=self._ANECDOCTOR_USE_KG_YAML)
        else:
            self._system_prompt_template = self._load_prompt_from_yaml(yaml_filename=self._ANECDOCTOR_USE_FEWSHOT_YAML)

    def _validate_context(self, *, context: AnecdoctorAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (AnecdoctorAttackContext): The attack context to validate.

        Raises:
            ValueError: If the context is invalid.
        """

        if not context.content_type:
            raise ValueError("content_type must be provided in the context")

        if not context.language:
            raise ValueError("language must be provided in the context")

        if not context.evaluation_data:
            raise ValueError("evaluation_data cannot be empty")

    async def _setup_async(self, *, context: AnecdoctorAttackContext) -> None:
        """
        Set up the attack by preparing the system prompt.

        Args:
            context (AnecdoctorAttackContext): The attack context containing parameters.
        """
        self._logger.debug(f"Setting up Anecdoctor attack for objective: {context.objective}")

        context.conversation_id = str(uuid.uuid4())

        # Format the system prompt with language and content type
        system_prompt = self._system_prompt_template.format(language=context.language, type=context.content_type)

        # Configure the target with the system prompt
        self._objective_target.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=context.conversation_id,
            orchestrator_identifier=self.get_identifier(),
            labels=context.memory_labels,
        )

    async def _perform_attack_async(self, *, context: AnecdoctorAttackContext) -> AttackResult:
        """
        Execute the Anecdoctor attack.

        Args:
            context (AnecdoctorAttackContext): The attack context containing all parameters.

        Returns:
            AttackResult: The result containing the generated misinformation content.
        """
        self._logger.info(f"Starting Anecdoctor attack with objective: {context.objective}")
        self._logger.debug(f"Using knowledge graph: {self._processing_model is not None}")
        self._logger.debug(f"Language: {context.language}, Content type: {context.content_type}")

        # Step 1: Prepare formatted examples (with or without KG extraction)
        formatted_examples = await self._prepare_examples_async(context=context)

        # Step 2: Send the prepared examples to the target model
        response = await self._send_examples_to_target_async(formatted_examples=formatted_examples, context=context)

        # Step 3: Evaluate the response if scoring is configured
        score = await self._evaluate_response_if_configured_async(response=response, objective=context.objective)

        # Step 4: Build and return the attack result
        return self._build_attack_result(response=response, score=score, context=context)

    async def _teardown_async(self, *, context: AnecdoctorAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to clean up for this attack
        pass

    async def _prepare_examples_async(self, *, context: AnecdoctorAttackContext) -> str:
        """
        Prepare the formatted examples, either directly or via knowledge graph extraction.

        Args:
            context (AnecdoctorAttackContext): The attack context.

        Returns:
            str: The formatted examples ready to send to the target.
        """
        if self._processing_model:
            # Extract knowledge graph from examples
            return await self._extract_knowledge_graph_async(context=context)
        else:
            # Use few-shot examples directly
            return self._format_few_shot_examples(evaluation_data=context.evaluation_data)

    async def _send_examples_to_target_async(
        self, *, formatted_examples: str, context: AnecdoctorAttackContext
    ) -> Optional[PromptRequestResponse]:
        """
        Send the formatted examples to the target model.

        Args:
            formatted_examples (str): The formatted examples to send.
            context (AnecdoctorAttackContext): The attack context.

        Returns:
            Optional[PromptRequestResponse]: The response from the target, or None if failed.
        """
        # Create seed prompt group
        prompt_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=formatted_examples,
                    data_type="text",
                )
            ]
        )

        # Send to target model with converters
        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_group,
            target=self._objective_target,
            conversation_id=context.conversation_id,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters,
            labels=context.memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

    async def _evaluate_response_if_configured_async(
        self, *, response: Optional[PromptRequestResponse], objective: str
    ) -> Optional[Score]:
        """
        Evaluate the response using the objective scorer if configured.

        Args:
            response (Optional[PromptRequestResponse]): The response to evaluate.
            objective (str): The attack objective.

        Returns:
            Optional[Score]: The score from the objective scorer, or None if not configured.
        """
        if not response or not self._objective_scorer:
            return None

        return await self._evaluate_response_async(response=response, objective=objective)

    def _build_attack_result(
        self, *, response: Optional[PromptRequestResponse], score: Optional[Score], context: AnecdoctorAttackContext
    ) -> AttackResult:
        """
        Build the final attack result based on the response and score.

        Args:
            response (Optional[PromptRequestResponse]): The response from the target.
            score (Optional[Score]): The objective score if available.
            context (AnecdoctorAttackContext): The attack context.

        Returns:
            AttackResult: The complete attack result.
        """
        outcome, outcome_reason = self._determine_attack_outcome(response=response, score=score)

        return AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=1,
        )

    def _determine_attack_outcome(
        self, *, response: Optional[PromptRequestResponse], score: Optional[Score]
    ) -> Tuple[AttackOutcome, Optional[str]]:
        """
        Determine the outcome of the attack based on the response and score.

        Args:
            response (Optional[PromptRequestResponse]): The response from the target.
            score (Optional[Score]): The objective score if scoring was configured.

        Returns:
            Tuple[AttackOutcome, Optional[str]]: A tuple of (outcome, outcome_reason).
        """
        if not self._objective_scorer:
            # No scorer configured, determine based on response only
            if response:
                return AttackOutcome.SUCCESS, "Successfully generated content based on examples"
            else:
                return AttackOutcome.FAILURE, "Failed to generate content (no response received)"

        # Scorer is configured, use it to determine outcome
        if score and score.get_value():
            return AttackOutcome.SUCCESS, "Objective achieved according to scorer"
        elif response:
            return AttackOutcome.FAILURE, "Response generated but objective not achieved according to scorer"
        else:
            return AttackOutcome.FAILURE, "Failed to generate content (no response received)"

    async def _evaluate_response_async(self, *, response: PromptRequestResponse, objective: str) -> Optional[Score]:
        """
        Evaluate the response against the objective using the configured scorers.

        Args:
            response (PromptRequestResponse): The response from the model.
            objective (str): The natural-language description of the attack's objective.

        Returns:
            Optional[Score]: The score from the objective scorer if configured.
        """
        scoring_results = await Scorer.score_response_with_objective_async(
            response=response,
            auxiliary_scorers=self._auxiliary_scorers,
            objective_scorers=[self._objective_scorer] if self._objective_scorer else None,
            role_filter="assistant",
            task=objective,
        )

        objective_scores = scoring_results["objective_scores"]
        if not objective_scores:
            return None

        return objective_scores[0]

    def _load_prompt_from_yaml(self, *, yaml_filename: str) -> str:
        """
        Load a prompt template from a YAML file.

        Args:
            yaml_filename (str): Name of the YAML file to load.

        Returns:
            str: The prompt template string from the 'value' key.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        prompt_path = Path(DATASETS_PATH, self._ANECDOCTOR_PROMPT_PATH, yaml_filename)
        prompt_data = prompt_path.read_text(encoding="utf-8")
        yaml_data = yaml.safe_load(prompt_data)
        return yaml_data["value"]

    def _format_few_shot_examples(self, *, evaluation_data: List[str]) -> str:
        """
        Format the evaluation data as few-shot examples.

        Args:
            evaluation_data (List[str]): The evaluation data to format.

        Returns:
            str: Formatted string with examples prefixed by "### examples".
        """
        return "### examples\n" + "\n".join(evaluation_data)

    async def _extract_knowledge_graph_async(self, *, context: AnecdoctorAttackContext) -> str:
        """
        Extract a knowledge graph from the evaluation data using the processing model.

        Args:
            context (AnecdoctorAttackContext): The attack context containing evaluation data.

        Returns:
            str: The extracted knowledge graph as a formatted string.

        Raises:
            RuntimeError: If knowledge graph extraction fails.
        """
        # Processing model is guaranteed to exist when this method is called
        assert self._processing_model is not None

        self._logger.debug("Extracting knowledge graph from evaluation data")

        # Load and format the KG extraction prompt

        kg_system_prompt = self._kg_prompt_template.format(language=context.language)

        # Create a separate conversation ID for KG extraction
        kg_conversation_id = str(uuid.uuid4())

        # Set system prompt on processing model
        self._processing_model.set_system_prompt(
            system_prompt=kg_system_prompt,
            conversation_id=kg_conversation_id,
            orchestrator_identifier=self.get_identifier(),
            labels=self._memory_labels,
        )

        # Format examples for KG extraction
        formatted_examples = self._format_few_shot_examples(evaluation_data=context.evaluation_data)

        # Create seed prompt group
        kg_prompt_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=formatted_examples,
                    data_type="text",
                )
            ]
        )

        # Send to processing model with converters
        kg_response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=kg_prompt_group,
            target=self._processing_model,
            conversation_id=kg_conversation_id,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters,
            labels=self._memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        if not kg_response:
            raise RuntimeError("Failed to extract knowledge graph: no response from processing model")

        return kg_response.get_value()

    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        content_type: str,
        language: str,
        evaluation_data: List[str],
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackResult:
        """
        Execute the attack asynchronously.

        Args:
            objective (str): The natural-language description of the attack's objective.
            content_type (str): The content type of the attack, e.g. "viral tweet".
            language (str): The language of the attack, e.g. "english".
            evaluation_data (List[str]): The evaluation data for the attack.
            memory_labels (Optional[dict[str, str]]): Memory labels for the attack.
            **kwargs: Additional keyword arguments.

        Returns:
            AttackResult: The result of the attack.
        """
        ...

    @overload
    async def execute_async(
        self,
        *,
        content_type: str,
        language: str,
        evaluation_data: List[str],
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackResult:
        """
        Execute the attack asynchronously.

        Args:
            content_type (str): The content type of the attack, e.g. "viral tweet".
            language (str): The language of the attack, e.g. "english".
            evaluation_data (List[str]): The evaluation data for the attack.
            memory_labels (Optional[dict[str, str]]): Memory labels for the attack.
            **kwargs: Additional keyword arguments.

        Returns:
            AttackResult: The result of the attack.
        """
        ...

    @overload
    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResult: ...

    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResult:
        """
        Execute the attack asynchronously.

        Args:
            objective (Optional[str]): The objective of the attack.
            content_type (str): The content type of the attack, e.g. "viral tweet".
            language (str): The language of the attack, e.g. "english".
            evaluation_data (List[str]): The evaluation data for the attack.
            memory_labels (Optional[dict[str, str]]): Memory labels for the attack.
            **kwargs: Additional keyword arguments.

        Returns:
            AttackResult: The result of the attack.
        """
        objective = get_kwarg_param(
            kwargs=kwargs,
            param_name="objective",
            required=False,
            expected_type=str,
        )

        content_type = get_kwarg_param(
            kwargs=kwargs,
            param_name="content_type",
            required=True,
            expected_type=str,
        )

        language = get_kwarg_param(
            kwargs=kwargs,
            param_name="language",
            required=True,
            expected_type=str,
        )

        evaluation_data = get_kwarg_param(
            kwargs=kwargs,
            param_name="evaluation_data",
            required=True,
            expected_type=list,
        )

        objective = objective or f"Generate a {content_type} in {language} using misinformation techniques"

        return await super().execute_async(
            objective=objective,
            content_type=content_type,
            language=language,
            evaluation_data=evaluation_data,
            **kwargs,
        )
