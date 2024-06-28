import json 
import requests
from typing import List, Dict
import csv


def fetch_examples(source: str, source_type: str = 'repository', file_type: str = 'json') -> List[Dict[str, str]]:
    """
        Fetch examples from a specified source.

        Args:
            source (str): The source from which to fetch examples.
            source_type (str): The type of source ('repository' or 'file'). Defaults to 'repository'.
            file_type (str): The type of file ('json' or 'csv'). Defaults to 'json'.

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
            examples = response.json()
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
        else:
            raise ValueError("Invalid file_type. Expected 'json' or 'csv'.")
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