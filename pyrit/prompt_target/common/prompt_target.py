# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Any, List, Optional

from pyrit.identifiers import Identifiable, TargetIdentifier
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Message, PromptDataType

logger = logging.getLogger(__name__)


class PromptTarget(Identifiable[TargetIdentifier]):
    """
    Abstract base class for prompt targets.

    A prompt target is a destination where prompts can be sent to interact with various services,
    models, or APIs. This class defines the interface that all prompt targets must implement.
    """

    _memory: MemoryInterface

    #: A list of PromptConverters that are supported by the prompt target.
    #: An empty list implies that the prompt target supports all converters.
    supported_converters: List[Any]

    #: Set of supported input modality combinations.
    #: Each frozenset represents a valid combination of modalities that can be sent together.
    #: For example: {frozenset(["text"]), frozenset(["text", "image_path"])}
    #: means the target supports either text-only OR text+image combinations.
    SUPPORTED_INPUT_MODALITIES: set[frozenset[PromptDataType]] = {frozenset(["text"])}

    #: Set of supported output modality combinations.
    #: Each frozenset represents a valid combination of modalities that can be returned.
    #: Most targets currently only support text output.
    SUPPORTED_OUTPUT_MODALITIES: set[frozenset[PromptDataType]] = {frozenset(["text"])}

    _identifier: Optional[TargetIdentifier] = None

    def __init__(
        self,
        verbose: bool = False,
        max_requests_per_minute: Optional[int] = None,
        endpoint: str = "",
        model_name: str = "",
        underlying_model: Optional[str] = None,
    ) -> None:
        """
        Initialize the PromptTarget.

        Args:
            verbose (bool): Enable verbose logging. Defaults to False.
            max_requests_per_minute (int, Optional): Maximum number of requests per minute.
            endpoint (str): The endpoint URL. Defaults to empty string.
            model_name (str): The model name. Defaults to empty string.
            underlying_model (str, Optional): The underlying model name (e.g., "gpt-4o") for
                identification purposes. This is useful when the deployment name in Azure differs
                from the actual model. If not provided, `model_name` will be used for the identifier.
                Defaults to None.
        """
        self._memory = CentralMemory.get_memory_instance()
        self._verbose = verbose
        self._max_requests_per_minute = max_requests_per_minute
        self._endpoint = endpoint
        self._model_name = model_name
        self._underlying_model = underlying_model

        if self._verbose:
            logging.basicConfig(level=logging.INFO)

    @abc.abstractmethod
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Send a normalized prompt async to the prompt target.

        Returns:
            list[Message]: A list of message responses. Most targets return a single message,
                but some (like response target with tool calls) may return multiple messages.
        """

    @abc.abstractmethod
    def _validate_request(self, *, message: Message) -> None:
        """
        Validate the provided message.

        Args:
            message: The message to validate.
        """

    def input_modality_supported(self, modalities: set[PromptDataType]) -> bool:
        """
        Check if a specific combination of input modalities is supported.
        
        Args:
            modalities: Set of modality types to check (e.g., {"text", "image_path"})
            
        Returns:
            True if this exact combination is supported, False otherwise
        """
        modalities_frozen = frozenset(modalities)
        return modalities_frozen in self.SUPPORTED_INPUT_MODALITIES

    def output_modality_supported(self, modalities: set[PromptDataType]) -> bool:
        """
        Check if a specific combination of output modalities is supported.
        Most targets only support text output currently.
        
        Args:
            modalities: Set of modality types to check
            
        Returns:
            True if this exact combination is supported, False otherwise
        """
        modalities_frozen = frozenset(modalities)
        return modalities_frozen in self.SUPPORTED_OUTPUT_MODALITIES

    async def verify_actual_capabilities(self) -> set[frozenset[PromptDataType]]:
        """
        Verify what modalities this target actually supports at runtime.
        
        This optional verification tests the target with minimal requests to determine
        actual capabilities, which may be a subset of the static API declarations.
        
        Returns:
            Set of actually supported input modality combinations
            
        Example:
            # Check what a specific OpenAI model actually supports
            actual = await target.verify_actual_capabilities()
            # Returns: {frozenset(["text"])} or {frozenset(["text"]), frozenset(["text", "image_path"])}
        """
        from pyrit.common.modality_verification import verify_target_capabilities
        return await verify_target_capabilities(self)

    def set_model_name(self, *, model_name: str) -> None:
        """
        Set the model name for this target.

        Args:
            model_name (str): The model name to set.
        """
        self._model_name = model_name

    def dispose_db_engine(self) -> None:
        """
        Dispose database engine to release database connections and resources.
        """
        self._memory.dispose_engine()

    def _create_identifier(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        target_specific_params: Optional[dict[str, Any]] = None,
    ) -> TargetIdentifier:
        """
        Construct the target identifier.

        Subclasses should call this method in their _build_identifier() implementation
        to set the identifier with their specific parameters.

        Args:
            temperature (Optional[float]): The temperature parameter for generation. Defaults to None.
            top_p (Optional[float]): The top_p parameter for generation. Defaults to None.
            target_specific_params (Optional[dict[str, Any]]): Additional target-specific parameters
                that should be included in the identifier. Defaults to None.

        Returns:
            TargetIdentifier: The identifier for this prompt target.
        """
        # Determine the model name to use
        model_name = ""
        if self._underlying_model:
            model_name = self._underlying_model
        elif self._model_name:
            model_name = self._model_name

        return TargetIdentifier(
            class_name=self.__class__.__name__,
            class_module=self.__class__.__module__,
            class_description=" ".join(self.__class__.__doc__.split()) if self.__class__.__doc__ else "",
            identifier_type="instance",
            endpoint=self._endpoint,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_requests_per_minute=self._max_requests_per_minute,
            target_specific_params=target_specific_params,
        )

    def _build_identifier(self) -> TargetIdentifier:
        """
        Build the identifier for this target.

        Subclasses can override this method to call _create_identifier() with
        their specific parameters (temperature, top_p, target_specific_params).

        The base implementation calls _create_identifier() with no parameters,
        which works for targets that don't have model-specific settings.

        Returns:
            TargetIdentifier: The identifier for this prompt target.
        """
        return self._create_identifier()
