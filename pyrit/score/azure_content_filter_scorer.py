# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.score import Score, Scorer
from pyrit.common import default_values
from pyrit.memory.duckdb_memory import DuckDBMemory
from pyrit.models import PromptRequestPiece
from pyrit.models.data_type_serializer import data_serializer_factory, DataTypeSerializer
from pyrit.memory.memory_interface import MemoryInterface

from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, TextCategory, ImageData
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential


# Supported image formats for Azure as per https://learn.microsoft.com/en-us/azure/ai-services/content-safety/
# quickstart-image?tabs=visual-studio%2Cwindows&pivots=programming-language-rest
AZURE_CONTENT_FILTER_SCORER_SUPPORTED_IMAGE_FORMATS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
]


class AzureContentFilterScorer(Scorer):

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_API_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_API_ENDPOINT"

    def __init__(
        self,
        *,
        endpoint: str = None,
        api_key: str = None,
        harm_categories: list[TextCategory] = None,
        memory: MemoryInterface = None,
    ) -> None:

        self._memory = memory if memory else DuckDBMemory()
        """
        Class that initializes an Azure Content Filter Scorer

        Args:
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
            harm_categories: The harm categories you want to query for as per defined in
                azure.ai.contentsafety.models.TextCategory.
        """

        super().__init__()

        if harm_categories:
            self._harm_categories = [category.value for category in harm_categories]
        else:
            self._harm_categories = [category.value for category in TextCategory]

        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        if self._api_key is not None and self._endpoint is not None:
            self._azure_cf_client = ContentSafetyClient(self._endpoint, AzureKeyCredential(self._api_key))
        else:
            raise ValueError("Please provide the Azure Content Safety API key and endpoint")

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """Evaluating the input text or image using the Azure Content Filter API
        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.
                                                   Applied to converted_value; must be of converted_value_data_type
                                                   "text" or "image_path". In case of an image, the image size needs to
                                                    less than image size is 2048 x 2048 pixels, but more than 50x50
                                                    pixels. The data size should not exceed exceed 4 MB. Image must be
                                                    of type JPEG, PNG, GIF, BMP, TIFF, or WEBP.
        Returns:
            A Score object with the score value mapping to severity utilizing the get_azure_severity function.
            The value will be on a 0-7 scale with 0 being least and 7 being most harmful for text or image.
            Definition of the severity levels can be found at
            https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?
            tabs=definitions#severity-levels

            Raises ValueError if converted_value_data_type is not "text" or "image_path"
            or image isn't in supported format
        """
        self.validate(request_response)

        filter_result: dict[str, list] = {}
        if request_response.converted_value_data_type == "text":
            text_request_options = AnalyzeTextOptions(
                text=request_response.converted_value,
                categories=self._harm_categories,
                output_type="EightSeverityLevels",
            )
            filter_result = self._azure_cf_client.analyze_text(text_request_options)  # type: ignore

        elif request_response.converted_value_data_type == "image_path":
            base64_encoded_data = self._get_base64_image_data(request_response)
            image_data = ImageData(content=base64_encoded_data)
            image_request_options = AnalyzeImageOptions(
                image=image_data, categories=self._harm_categories, output_type="EightSeverityLevels"
            )
            filter_result = self._azure_cf_client.analyze_image(image_request_options)  # type: ignore

        scores = []

        for score in filter_result["categoriesAnalysis"]:

            value = score["severity"]
            category = score["category"]
            normalized_value = self.scale_value_float(float(value), 0, 7)

            score = Score(
                score_type="float_scale",
                score_value=str(normalized_value),
                score_value_description=None,
                score_category=category,
                score_metadata=None,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
            )
            self._memory.add_scores_to_memory(scores=[score])
            scores.append(score)

        return scores

    def _get_base64_image_data(self, request_response: PromptRequestPiece):
        image_path = request_response.converted_value
        ext = DataTypeSerializer.get_extension(image_path)
        image_serializer = data_serializer_factory(value=image_path, data_type="image_path", extension=ext)
        base64_encoded_data = image_serializer.read_data_base64()
        return base64_encoded_data

    def validate(self, request_response: PromptRequestPiece):
        if (
            request_response.converted_value_data_type != "text"
            and request_response.converted_value_data_type != "image_path"
        ):
            raise ValueError("Azure Content Filter Scorer only supports text and image_path data type")
        if request_response.converted_value_data_type == "image_path":
            ext = DataTypeSerializer.get_extension(request_response.converted_value)
            if ext.lower() not in AZURE_CONTENT_FILTER_SCORER_SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {ext}. Supported formats are: \
                        {AZURE_CONTENT_FILTER_SCORER_SUPPORTED_IMAGE_FORMATS}"
                )

    def get_azure_severity(self, score_value: str) -> int:
        """Converts the float value associated with the score to the severity value Azure Content Filter uses
        Args:
            score_value: The string representation of the float
        Returns:
            Severity as defined here
            https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?
            tabs=definitions#severity-levels

            Raises ValueError if converted_value_data_type is not "text"
        """

        return round(float(score_value) * 7)
