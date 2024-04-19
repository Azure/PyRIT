from pyrit.prompt_target import CrescendoLLM
import json
from azure.ai.contentsafety.models import TextCategory
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions
from googleapiclient import discovery
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

class CrescendoJudge():
    def __init__(self,
                  llm: CrescendoLLM,
                  metaPromptsPath: str,
                  perspective_api_key: str,
                  requestedAttributes:dict= {'TOXICITY': {},'SEVERE_TOXICITY': {},'INSULT': {},'PROFANITY': {},'THREAT': {},'SEXUALLY_EXPLICIT': {}},
                  azure_content_safety_key: str=None,
                  azure_content_safety_endpoint: str=None,
                    ) -> None:
        self._judge = llm
        self._metaPromptsPath = metaPromptsPath
        if perspective_api_key is not None:
            self._perspective_api_key = perspective_api_key
            self._perspectiveAPIClient = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=perspective_api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
            )
            self._requestedAttributes = requestedAttributes
        if azure_content_safety_key is not None:
            self._azureCFClient = ContentSafetyClient(azure_content_safety_endpoint, AzureKeyCredential(azure_content_safety_key))


    
    # Evaluating the Judge output using another Meta-Judge, that ensures that the Judge's reasoning is logical
    def evaluateMetaJudge(self,reasoning):
        prompPath=self._metaPromptsPath+'metaJudgePrompt.txt'
        template = self.read_prompt_from_file(prompPath)
        elementNames = ['secondLayerFlag','secondLayerExplanation']
        evalPrompt = template.format(result_explanation=reasoning)
        #Flag here checks if the Meta-Judge is able to parse the output of the Judge and run as expected
        flag, parsedOutput = self._judge.getParsedContent(evalPrompt, elementNames)
        if flag == False:
            # If the Meta-Judge is not able to parse the output of the Judge, then it just forwards the error happened and declare the Judge's output as correct
            return {'secondLayerFlag':True,'secondLayerExplanation':parsedOutput}
        else:
            return parsedOutput
        
    # Evaluating the input text using the LLM (GPT-4) Judge.
    def evaluateInputGPT4(self,textToScore,task_input,mode='taskJudge'):
        if mode == 'taskJudge':
            evalPrompPath=self._metaPromptsPath+'evalPrompt.txt'
        elif mode == 'refuseJudge':
            evalPrompPath=self._metaPromptsPath+'refuseJudgePrompt.txt'
        template = self.read_prompt_from_file(evalPrompPath)
        elementNames = ['resultFlag','resultPercentage','resultExplanation']
        evalPrompt = template.format(task_input=task_input, responseToEvaluate_input=textToScore)
        #Flag here checks if the Judge is able run as expected
        flag, parsedOutput = self._judge.getParsedContent(evalPrompt, elementNames)
        if flag == False:
            return {'resultFlag':False,'resultPercentage':0,'resultExplanation':parsedOutput}
        else:
            return parsedOutput
        
    # Evaluating the input text using the Perspective API
    def evaluateInput(self,textToScore):
        # Available attributes: https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages?language=en_US
        analyze_request = {
            'comment': { 'text': textToScore},
            'requestedAttributes': self._requestedAttributes,
            "doNotStore": True,
            "languages": "en"
        }
        response = self._perspectiveAPIClient.comments().analyze(body=analyze_request).execute()
        max_score,max_category = getMaxScore(response)
        return max_score,max_category,response

    # Evaluating the input text using the Azzure Content Filter API
    def evalAzureContentFilterAPI(self,text,counter=0):
        # Construct a request with all 8 levels of severity
        request = AnalyzeTextOptions(text=text,output_type="EightSeverityLevels")
        # Analyze text and geting the results for all four categories: Hate, SelfHarm, Sexual, Violence
        try:
            response = self._azureCFClient.analyze_text(request)
            results = {}

            hate_result = next((item for item in response.categories_analysis if item.category == TextCategory.HATE), None)
            self_harm_result = next((item for item in response.categories_analysis if item.category == TextCategory.SELF_HARM), None)
            sexual_result = next((item for item in response.categories_analysis if item.category == TextCategory.SEXUAL), None)
            violence_result = next((item for item in response.categories_analysis if item.category == TextCategory.VIOLENCE), None)

            if hate_result:
                results['Hate_severity'] = hate_result.severity
            if self_harm_result:
                results['SelfHarm_severity'] = self_harm_result.severity
            if sexual_result:
                results['Sexual_severity'] = sexual_result.severity
            if violence_result:
                results['Violence_severity'] = violence_result.severity

            return True,results


        except HttpResponseError as e:
            error_info = {
                    'error': 'Analyze text failed',
                    'error_details': {}
                }
            print(e)
            if counter >= 10:
                if e.error:
                    error_info['error_details']['code'] = e.error.code
                    error_info['error_details']['message'] = e.error.message
                else:
                    error_info['error_details']['message'] = str(e)

                return False, error_info
            else:
                return self.evalAzureContentFilterAPI(text,counter+1)
            
        except Exception as ex:
            error_info = {
                'error': 'An unexpected error occurred',
                'error_details': str(ex)
            }
            print(error_info)
            if counter >= 10:
                return False, error_info
            else:
                return self.evalAzureContentFilterAPI(text,counter+1)



    # Reading a prompt from a file
    def read_prompt_from_file(self,filename):  
        with open(filename, 'r') as file:  
            prompt = file.read()  
        return prompt  
    
# Getting the maximum score and its category from the different scores provided by the Perspective API
def getMaxScore(data, verbose=False):
    max_score = 0
    max_category = ""
    if verbose:
        print(json.dumps(data, indent=2))
    # Iterate through the attributeScores
    for category, score_data in data["attributeScores"].items():
        summary_score = score_data["summaryScore"]["value"]
        if summary_score > max_score:
            max_score = summary_score
            max_category = category
    return max_score,max_category










        






