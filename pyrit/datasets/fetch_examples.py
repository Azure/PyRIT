import json 
import requests
from typing import List, Dict

class DatasetFetcher:
    def import_examples(self, source: str, source_type: str = 'repository') -> List[Dict[str, str]]:
        """
        Import examples from a specified source.

        Args:
            source (str): The source from which to import examples.
            source_type (str): The type of source ('repository' or 'user'). Defaults to 'repository'.

        Returns:
            List[Dict[str, str]]: A list of examples.

        Example usage:
            fetch_many_shot_jailbreaking_examples(
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
        elif source_type == 'user':
            # Load examples from a user-provided file (e.g., JSON format)
            with open(source, 'r') as file:
                examples = json.load(file)
        else:
            raise ValueError("Invalid source_type. Expected 'repository' or 'user'.")

        return examples