# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.interfaces import SupportTextClassification
from pyrit.models import Score

from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.ai.contentsafety.models import TextCategory
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential


class AzureContentFilter(SupportTextClassification):
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

    def score_text(self, text: str) -> Score:
        """Evaluating the input text using the Azure Content Filter API
        Args:
            text: The text to evaluate
        Returns:
            A Score object with the score value on a 0-7 scale with 0 being least and 7 being most harmful.
            Definition of the severity levels can be found at
            https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?
            tabs=definitions#severity-levels
        """
        request = AnalyzeTextOptions(text=text, output_type="EightSeverityLevels")
        # Analyze text and geting the results for the category specified

        response = self._azureCFClient.analyze_text(request)
        result = next((item for item in response.categories_analysis if item.category == self._harm_category), None)
        return Score(score_type="int", score_value=result.severity, score_description=result.category)
