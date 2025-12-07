# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeImageOptions,
    AnalyzeTextOptions,
    ImageData,
    TextCategory,
)
from azure.core.credentials import AzureKeyCredential

from pyrit.auth.azure_auth import AzureAuth, get_default_scope
from pyrit.common import default_values
from pyrit.models import (
    DataTypeSerializer,
    MessagePiece,
    Score,
    data_serializer_factory,
)
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


class AzureContentFilterScorer(FloatScaleScorer):
    """
    A scorer that uses Azure Content Safety API to evaluate text and images for harmful content.

    This scorer analyzes content across multiple harm categories (hate, self-harm, sexual, violence)
    and returns a score for each category in the range [0, 1], where higher scores indicate
    more severe content. Supports both text and image inputs.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(
        supported_data_types=["text", "image_path"],
    )

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_API_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_API_ENDPOINT"

    def __init__(
        self,
        *,
        endpoint: Optional[str | None] = None,
        api_key: Optional[str | None] = None,
        use_entra_auth: bool = False,
        harm_categories: Optional[list[TextCategory]] = None,
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        """
        Initialize an Azure Content Filter Scorer.

        Args:
            endpoint (Optional[str | None]): The endpoint URL for the Azure Content Safety service.
                Defaults to the `ENDPOINT_URI_ENVIRONMENT_VARIABLE` environment variable.
            api_key (Optional[str | None]): The API key for accessing the Azure Content Safety service
                (only if you're not using Entra authentication). Defaults to the `API_KEY_ENVIRONMENT_VARIABLE`
                environment variable.
            use_entra_auth (bool): Whether to use Entra authentication. Defaults to False.
            harm_categories (Optional[list[TextCategory]]): The harm categories you want to query for as
                defined in azure.ai.contentsafety.models.TextCategory. If not provided, defaults to all categories.
            validator (Optional[ScorerPromptValidator]): Custom validator for the scorer. Defaults to None.

        Raises:
            ValueError: If neither API key nor endpoint is provided, or if both are missing.
        """
        super().__init__(validator=validator or self._default_validator)

        if harm_categories:
            self._score_categories = [category.value for category in harm_categories]
        else:
            self._score_categories = [category.value for category in TextCategory]

        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint or ""
        )

        if not use_entra_auth:
            self._api_key = default_values.get_required_value(
                env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key or ""
            )
        else:
            if api_key:
                raise ValueError("Please specify either use_add_auth or api_key")
            else:
                self._api_key = None

        if self._api_key is not None and self._endpoint is not None:
            self._azure_cf_client = ContentSafetyClient(self._endpoint, AzureKeyCredential(self._api_key))
        elif use_entra_auth and self._endpoint is not None:
            azure_auth = AzureAuth(token_scope=get_default_scope(self._endpoint))
            self._azure_cf_client = ContentSafetyClient(self._endpoint, credential=azure_auth.azure_creds)
        else:
            raise ValueError("Please provide the Azure Content Safety endpoint")

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Evaluate the input text or image using the Azure Content Filter API.

        Args:
            message_piece (MessagePiece): The message piece containing the text or image to be scored.
                Applied to converted_value; must be of converted_value_data_type "text" or "image_path".
                In case of an image, the image size must be less than 2048 x 2048 pixels,
                but more than 50x50 pixels. The data size should not exceed 4 MB. Image must be
                of type JPEG, PNG, GIF, BMP, TIFF, or WEBP.
            objective (Optional[str]): The objective for scoring context. Currently not supported for this scorer.
                Defaults to None.

        Returns:
            list[Score]: A list of Score objects with score values mapping to severity utilizing the
                get_azure_severity function. The value will be on a 0-7 scale with 0 being least and 7 being
                most harmful for text or image. Definition of the severity levels can be found at
                https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=definitions#severity-levels

        Raises:
            ValueError: If converted_value_data_type is not "text" or "image_path" or image isn't in supported format.
        """
        filter_result: dict[str, list] = {}
        if message_piece.converted_value_data_type == "text":
            text_request_options = AnalyzeTextOptions(
                text=message_piece.converted_value,
                categories=self._score_categories,
                output_type="EightSeverityLevels",
            )
            filter_result = self._azure_cf_client.analyze_text(text_request_options)  # type: ignore

        elif message_piece.converted_value_data_type == "image_path":
            base64_encoded_data = await self._get_base64_image_data(message_piece)
            image_data = ImageData(content=base64_encoded_data)
            image_request_options = AnalyzeImageOptions(
                image=image_data, categories=self._score_categories, output_type="FourSeverityLevels"
            )
            filter_result = self._azure_cf_client.analyze_image(image_request_options)  # type: ignore

        scores = []

        for score in filter_result["categoriesAnalysis"]:

            value = score["severity"]
            category = score["category"]
            normalized_value = self.scale_value_float(float(value), 0, 7)

            # Severity as defined here
            # https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=definitions#severity-levels
            metadata: dict[str, str | int] = {"azure_severity": int(value)}

            score = Score(
                score_type="float_scale",
                score_value=str(normalized_value),
                score_value_description="",
                score_category=[category] if category else None,
                score_metadata=metadata,
                score_rationale="",
                scorer_class_identifier=self.get_identifier(),
                message_piece_id=message_piece.id,
                objective=objective,
            )
            scores.append(score)

        return scores

    async def _get_base64_image_data(self, message_piece: MessagePiece):
        image_path = message_piece.converted_value
        ext = DataTypeSerializer.get_extension(image_path)
        image_serializer = data_serializer_factory(
            category="prompt-memory-entries", value=image_path, data_type="image_path", extension=ext
        )
        base64_encoded_data = await image_serializer.read_data_base64()
        return base64_encoded_data
