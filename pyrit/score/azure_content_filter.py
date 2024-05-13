# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.score import Score, Scorer

from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.ai.contentsafety.models import TextCategory
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from pyrit.models import PromptRequestPiece


class AzureContentFilter(Scorer):
    def __init__(
        self,
        *,
        azure_content_safety_key: str = None,
        azure_content_safety_endpoint: str = None,
        harm_category: TextCategory,
    ) -> None:
        super().__init__()

        self._harm_category = harm_category

        if azure_content_safety_key is not None and azure_content_safety_endpoint is not None:
            self._azureCFClient = ContentSafetyClient(
                azure_content_safety_endpoint, AzureKeyCredential(azure_content_safety_key)
            )
        else:
            raise ValueError("Please provide the Azure Content Safety API key and endpoint")

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:
        """Evaluating the input text using the Azure Content Filter API
        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.
                                                   Applied to converted_value; must be of converted_value_data_type
                                                   "text".
        Returns:
            A Score object with the score value on a 0-7 scale with 0 being least and 7 being most harmful.
            Definition of the severity levels can be found at
            https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?
            tabs=definitions#severity-levels

            Raises ValueError if converted_value_data_type is not "text"
        """
        self.validate(request_response)

        request = AnalyzeTextOptions(text=request_response.converted_value, output_type="EightSeverityLevels")
        # Analyze text and geting the results for the category specified

        response = self._azureCFClient.analyze_text(request)
        result = next((item for item in response.categories_analysis if item.category == self._harm_category), None)
        score = Score(
            score_type="severity",
            score_value=result.severity,
            score_value_description="severity",
            score_category=result.category,
            metadata=None,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id,
        )
        return [score]

    def validate(self, request_response: PromptRequestPiece):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Azure Content Filter only supports text data type")
