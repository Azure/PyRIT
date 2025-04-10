from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import uuid

from pyrit.models.literals import PromptDataType
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration


@dataclass
class PromptValidationCriteria:
    """Criteria for validating normalizer requests"""
    multipart_allowed: bool = False
    allowed_data_types: List[PromptDataType] = None
    
    def __post_init__(self):
        if self.allowed_data_types is None:
            self.allowed_data_types = ["text"]


class PromptUtils:
    """
    Utility class for handling prompt normalization and validation.
    This class provides methods to build normalizer requests from prompts,
    validate them against certain criteria, and create normalizer requests
    with optional metadata and converters.
    """

    @staticmethod
    def build_normalizer_requests(
        *,
        prompts: List[str],
        prompt_type: PromptDataType = "text",
        converters: Optional[List[PromptConverter]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> List[NormalizerRequest]:
        """
        Build normalizer requests from the provided prompts.
        
        Args:
            prompts: The list of prompts to normalize
            batch_size: The maximum batch size for sending prompts
            memory_labels: Optional memory labels for the attack
            
        Returns:
            A list of NormalizerRequest objects
        """
        if not prompts:
            raise ValueError("No prompts provided")

        return [
            PromptUtils._create_normalizer_request(
                prompt_text=prompt,
                prompt_type=prompt_type,
                converters=converters or [],
                metadata=metadata,
                conversation_id=str(uuid.uuid4()),
            )
            for prompt in prompts
        ]
    
    @staticmethod
    def _create_normalizer_request(
        *,
        prompt_text: str,
        prompt_type: PromptDataType = "text",
        conversation_id: Optional[str] = None,
        converters: Optional[List[PromptConverter]] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ) -> NormalizerRequest:
        """
        Create a normalizer request for a prompt.
        
        Args:
            prompt_text: The text of the prompt
            prompt_type: The type of the prompt data
            converters: Optional list of prompt converters
            metadata: Optional metadata for the prompt

        Returns:
            A NormalizerRequest object
        """
        seed_prompt_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=prompt_text,
                    data_type=prompt_type,
                    metadata=metadata,
                )
            ]
        )

        converter_configurations = [PromptConverterConfiguration(
            converters=converters or []
        )]

        return NormalizerRequest(
            seed_prompt_group=seed_prompt_group,
            request_converter_configurations=converter_configurations,
            conversation_id=conversation_id or str(uuid.uuid4()),
        )
    
    @staticmethod
    def validate_normalizer_requests(
        *,
        requests: List[NormalizerRequest],
        criteria: PromptValidationCriteria,
    ) -> None:
        """
        Validate normalizer requests based on provided criteria.
        
        Args:
            requests: The list of normalizer requests to validate
            criteria: The validation criteria
            
        Raises:
            ValueError: If any request doesn't meet the criteria
        """
        if not requests:
            raise ValueError("No normalizer requests provided")

        for request in requests:
            # Check if multipart messages are allowed
            if not criteria.multipart_allowed and request.is_multipart():
                raise ValueError("Multi-part messages not supported")
                
            # Check that data types are supported
            for prompt in request.seed_prompt_group.prompts:
                if prompt.data_type not in criteria.allowed_data_types:
                    supported = ", ".join(criteria.allowed_data_types)
                    raise ValueError(f"Unsupported data type: {prompt.data_type}. Supported types: {supported}")