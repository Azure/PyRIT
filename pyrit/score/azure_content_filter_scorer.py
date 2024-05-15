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
        self, *, endpoint: str = None, api_key: str = None, harm_category: TextCategory, memory: MemoryInterface = None
    ) -> None:

        self._memory = memory if memory else DuckDBMemory()
        """
        Class that initializes an Azure Content Filter Scorer

        Args:
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
            harm_category: The harm category you want to query for as per defined in
                azure.ai.contentsafety.models.TextCategory.
        """

        super().__init__()

        self._harm_category = harm_category

        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        if self._api_key is not None and self._endpoint is not None:
            self._azureCFClient = ContentSafetyClient(self._endpoint, AzureKeyCredential(self._api_key))
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
            A Score object with the score value on a 0-7 scale with 0 being least and 7 being most harmful for text.
            A Score object with the score value on a 0,2,4,6 scale with 0 being least and 6 being most harmful for
            image.
            Definition of the severity levels can be found at
            https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?
            tabs=definitions#severity-levels

            Raises ValueError if converted_value_data_type is not "text"
        """
        self.validate(request_response)

        response = None
        if request_response.converted_value_data_type == "text":
            request = AnalyzeTextOptions(
                text=request_response.converted_value,
                categories=[self._harm_category],
                output_type="EightSeverityLevels",
            )
            # Analyze text and geting the results for the category specified
            response = self._azureCFClient.analyze_text(request)
        elif request_response.converted_value_data_type == "image_path":
            ext = DataTypeSerializer.get_extension(request_response.converted_value)
            image_serializer = data_serializer_factory(
                value=request_response.converted_value, data_type="image_path", extension=ext
            )
            base64_encoded_data = image_serializer.read_data_base64()

            image_data = ImageData(content=base64_encoded_data)
            request = AnalyzeImageOptions(
                image=image_data, categories=[self._harm_category], output_type="FourSeverityLevels"
            )
            # Analyze image and geting the results for the category specified
            response = self._azureCFClient.analyze_image(request)

        if response is not None:
            result = next((item for item in response.categories_analysis if item.category == self._harm_category), None)
            score = Score(
                score_type="severity",
                score_value=result.severity,
                score_value_description="severity",
                score_category=result.category,
                score_metadata=None,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
            )
            self._memory.add_scores_to_memory(scores=[score])
            return [score]
        else:
            return []

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
