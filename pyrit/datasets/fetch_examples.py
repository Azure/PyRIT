import json 
import requests
from typing import List, Dict
import csv
import pycountry
from pyrit.models import PromptDataset
import random


def fetch_examples(source: str, source_type: str = 'repository', file_type: str = 'json') -> List[Dict[str, str]]:
    """
        Fetch examples from a specified source.

        Args:
            source (str): The source from which to fetch examples.
            source_type (str): The type of source ('repository' or 'file'). Defaults to 'repository'.
            file_type (str): The type of file ('json', 'csv', or 'txt'). Defaults to 'json'.

        Returns:
            List[Dict[str, str]]: A list of examples.

        Example usage:
            examples = fetch_examples(
                source='https://github.com/KutalVolkan/many-shot-jailbreaking-dataset',
                source_type='repository'
            )
        
        Note:
            The dataset sources can be found at:
            - Original: https://github.com/centerforaisafety/HarmBench
            - Replicated: https://github.com/KutalVolkan/many-shot-jailbreaking-dataset
        """
    if source_type == 'repository':
        # Fetch examples from an external repository (e.g., via an API call)
        response = requests.get(source)
        if response.status_code == 200:
            if file_type == 'json':
                examples = response.json()
            elif file_type == 'csv':
                reader = csv.DictReader(response.text.splitlines())
                examples = [row for row in reader]
            elif file_type == 'txt':
                examples = [{'prompt': line} for line in response.text.splitlines()]
            else:
                raise ValueError("Invalid file_type. Expected 'json', 'csv', or 'txt'.")
            print("Examples fetched from repository:")
        else:
            raise Exception(f"Failed to fetch examples from repository. Status code: {response.status_code}")
    elif source_type == 'file':
        # Load examples from a user-provided file (e.g., JSON or CSV format)
        if file_type == 'json':
            with open(source, 'r') as file:
                examples = json.load(file)
        elif file_type == 'csv':
            with open(source, 'r') as file:
                reader = csv.DictReader(file)
                examples = [row for row in reader]
        elif file_type == 'txt':
                examples = [{'prompt': line} for line in response.text.splitlines()]
        else:
            raise ValueError("Invalid file_type. Expected 'json', 'csv', or 'txt'.")
    else:
        raise ValueError("Invalid source_type. Expected 'repository' or 'file'.")

    return examples



def fetch_many_shot_jailbreaking_examples(source: str, source_type: str = 'repository') -> List[Dict[str, str]]:
    """
    Fetch many-shot jailbreaking examples from a specified source.

    Args:
        source (str): The source from which to fetch examples.
        source_type (str): The type of source ('repository' or 'file'). Defaults to 'repository'.

    Returns:
        List[Dict[str, str]]: A list of many-shot jailbreaking examples.
    """
    return fetch_examples(source, source_type, file_type='json')


def fetch_seclists_bias_testing_examples(source: str, source_type: str = 'repository', file_type: str = 'txt') -> PromptDataset:
    """
    Fetch SecLists AI LLM Bias Testing examples from a specified source and create a PromptDataset.

    Args:
        source (str): The source from which to fetch examples.
        source_type (str): The type of source ('repository' or 'file'). Defaults to 'repository'.
        file_type (str): The type of file ('json', 'csv', or 'txt'). Defaults to 'txt'.

    Returns:
        PromptDataset: A PromptDataset containing the examples with placeholders replaced.
    """

    # Define placeholder values generation functions using pycountry and random.choice
    countries = list(pycountry.countries)
    subdivisions = list(pycountry.subdivisions)
    placeholder_values = {
        "Country": lambda: random.choice(countries).name,
        "Region": lambda: random.choice(subdivisions).name,
        "Nationality": lambda: random.choice(countries).name, # Same as Country, the target LLM should be able to understand the context
        "GENDER": lambda: random.choice(["male", "female", "non-binary"]),
        "SKIN_COLOR": lambda: random.choice(["white", "black", "brown", "yellow"])
    }

    examples = fetch_examples(source, source_type, file_type)
    
    filled_examples = []
    for example in examples:
        prompt = example['prompt']
        #print(f"Original Prompt: {prompt}")  # Debug statement 
        for placeholder, generator in placeholder_values.items():
            values_used = set()
            while f"[{placeholder}]" in prompt:
                value = generator()
                # Ensure the new value is not the same as the previous one
                while value in values_used:
                    value = generator()
                values_used.add(value)
                prompt = prompt.replace(f"[{placeholder}]", value, 1)
                #print(f"Replaced [{placeholder}] with {value}")  # Debug statement
        
        #print(f"Filled Prompt: {prompt}\n")  # Debug statement
        
        filled_examples.append(prompt)


    dataset = PromptDataset(
        name="SecLists Bias Testing Examples",
        description="A dataset of SecLists AI LLM Bias Testing examples with placeholders replaced.",
        harm_category="bias_testing",
        should_be_blocked=False,
        prompts=filled_examples
    )

    return dataset