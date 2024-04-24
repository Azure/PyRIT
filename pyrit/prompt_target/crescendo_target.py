# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from transformers import LlamaForCausalLM, AutoTokenizer
import time
from langchain_anthropic import ChatAnthropic
import google.generativeai as genai


class CrescendoLLM():

    def __init__(
        self,
        llmName: str,
        api_key: str,
        api_deployment_name: str,
        api_endpoint: str=None,
        api_version: str="2023-05-15",
        LlamaHuggingFace: bool = False,
        GeminiSafetySettings: list = None,
        temperature: float = 0.5,
        request_timeout: int = 120
    ) -> None:
        super().__init__(
        )
        self._llmName = llmName
        if llmName == "Llama-2-70b":
            self._LlamaHuggingFace = LlamaHuggingFace
            self._llm = getLlama2_70Instance(LlamaHuggingFace=LlamaHuggingFace,api_deployment_name=api_deployment_name)
        elif llmName == "GPT-4":
            self._llm = getAGPT4Instance(api_key=api_key,depoyment_name=api_deployment_name,api_endpoint=api_endpoint,api_version=api_version,temperature=temperature, request_timeout=request_timeout)
        elif llmName == "GPT-3.5":
            self._llm = getAGPT3Instance(api_key=api_key,depoyment_name=api_deployment_name,api_endpoint=api_endpoint,api_version=api_version,temperature=temperature, request_timeout=request_timeout)
        elif llmName == "Gemini":
            
            # Set the safety settings for Gemini
            if GeminiSafetySettings is None:
                self._GeminiSafetySettings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS",
                        "threshold": "BLOCK_NONE",
                    }
                ]
            else:
                 self._GeminiSafetySettings = GeminiSafetySettings
            self._llm = getGeminiPromptInstance(api_key=api_key,api_deployment_name=api_deployment_name)
        elif llmName == "Claude":
            self._llm = getClaudeInstance(api_key=api_key,api_deployment_name=api_deployment_name)
        else:
            raise ValueError("Invalid LLM name")

        # Prompt templates


    # Adapting the chat history to the Llama-2 format
    def langhchain_to_llama2(inputObject,Llama2Tokenizer):
        history = Llama2Tokenizer.eos_token
        for item in inputObject:
            if item.type == 'human':
                history += "[INST]" + item.content + "[/INST]"
            else:
                history += item.content + Llama2Tokenizer.eos_token
        tokens = Llama2Tokenizer(history, return_tensors="pt", add_special_tokens=False)
        return history, tokens['input_ids']

    # Parsing the output of the LLM
    def getParsedContent(self, template, elementNames): 
        llm = self._llm
        counter = 0
        # Trying to parse the output 5 times else returning an error
        while counter < 5:  
            counter = counter+ 1
            try:  
                
                flag,output = self.callLLM([HumanMessage(content=template)])
                if not flag:
                    print("Error: %s"%output)
                    return flag,output 
                
                parsed_output = json.loads(output.content)  
                parsed_elements = {}  
                
                for elementName in elementNames:  
                    parsedElement = parsed_output.get(elementName)  
                    if parsedElement is not None:  
                        parsed_elements[elementName] = parsedElement  
                    else:  
                        print(output.content)  
                        print(f"Error: '{elementName}' key not found in the output")  
                        break  
                else:  
                    return True,parsed_elements  
    
            except json.JSONDecodeError as e:  
                print(output.content)  
                print("Error: Incorrect JSON format", str(e))  
                continue  

        return False,'Error: Incorrect JSON format'

    # Adapting the chat history to the Gimini format
    def langchain_to_gemini(inputObject):
        history = [
            {
                "role": "user" if item.type == 'human' else "model",
                "parts": [{"text": item.content}],
            } 
            for item in inputObject
        ]
        return history

    # Calling the LLM. Adapting the input format depending on the LLM
    def callLLM(self, inputObject):
        llm = self._llm
        try:
            if type(llm) == genai.GenerativeModel:
                chat = llm.start_chat(history=[])
                chat.history = self.langchain_to_gemini(inputObject)[:-1]                
                currentPrompt = inputObject[len(inputObject)-1].content
                response = chat.send_message( currentPrompt, safety_settings=self._GeminiSafetySettings )
                output = AIMessage(content=response.text)
            elif type(llm) == LlamaForCausalLM or type(llm) == LLM:
                prompt, promptTokens = self.langhchain_to_llama2(inputObject)
                if self._LlamaHuggingFace is True:
                    promptTokens = promptTokens.to(llm.device)
                    output_tokens = llm.generate(promptTokens, max_new_tokens=1000 )
                    text = Llama2Tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                    last_inst = text.rfind("[/INST]")
                    if last_inst != -1:
                        output = AIMessage(content=text[last_inst+7:])
                    else:
                        raise Exception("No [/INST] found in output")                
                else: 
                    samplingParmams = SamplingParams( max_tokens=1000 )
                    outputObject = llm.generate(prompt[-4000:], samplingParmams)
                    text = outputObject[0].outputs[0].text
                    output = AIMessage(content=text)
            else:
                output = llm.invoke(inputObject)
            return True, output
        except Exception as e:
            if '429' in str(e) or 'exceeded call rate limit' in str(e):
                print('429 error, waiting 10 seconds')
                time.sleep(10)
                return self.callLLM(llm, inputObject)
            if '529' in str(e):
                print('529 error, waiting 60 seconds')
                time.sleep(60)
                return self.callLLM(llm, inputObject)
            if 'error' in str(e):
                print('unknown error, waiting 60 seconds')
                print(str(e))
                time.sleep(60)
                return self.callLLM(llm, inputObject)
            return False, str((e))
        
    def read_prompt_from_file(self,filename):  
        with open(filename, 'r') as file:  
            prompt = file.read()  
        return prompt  
# Initializing a GPT-3 instance using Azure OpenAI API
def getAGPT3Instance(api_key,depoyment_name='gpt35turbo',api_endpoint=None,temperature=0.5, request_timeout=60,api_version = "2023-05-15"):
    gpt3_api_key = api_key
    gpt3_api_endpoint = api_endpoint
    return AzureChatOpenAI(deployment_name=depoyment_name,api_key=gpt3_api_key,api_version = api_version, 
            azure_endpoint=gpt3_api_endpoint,temperature=temperature, request_timeout=request_timeout)

# Initializing a Llama-2 70b instance
def getLlama2_70Instance(LlamaHuggingFace,api_deployment_name="meta-llama/Llama-2-70b-chat-hf"):
    global Llama2Tokenizer
    Llama2Tokenizer = AutoTokenizer.from_pretrained(api_deployment_name)
    if LlamaHuggingFace is True:
        Llama70bModel = AutoModelForCausalLM.from_pretrained(api_deployment_name, device_map="auto" )   
    else:
        Llama70bModel = LLM(api_deployment_name, tensor_parallel_size=4)
    return Llama70bModel


# Initializing a GPT-4 instance using Azure OpenAI API
def getAGPT4Instance(api_key,depoyment_name='gpt-4',api_endpoint=None,temperature=0.5, request_timeout=120,api_version = "2023-05-15"):
    gpt4_api_key = api_key
    gpt4_api_depoyment_name = depoyment_name
    gpt4_api_endpoint = api_endpoint
    return AzureChatOpenAI(azure_deployment=gpt4_api_depoyment_name,api_key=gpt4_api_key,api_version = api_version,
            azure_endpoint=gpt4_api_endpoint,temperature=temperature, request_timeout=request_timeout)

# Initializing a Gemini instance 
def getGeminiPromptInstance(api_key,api_deployment_name='gemini-pro'):
    GOOGLE_API_KEY=  api_key#get_config_value("GOOGLE_API_GEMINI")#os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    GeminiProModel = genai.GenerativeModel(api_deployment_name)
    return GeminiProModel

def getClaudeInstance(api_key,api_deployment_name='claude-3-opus-20240229'):
    return ChatAnthropic(api_key=api_key,model=api_deployment_name)

