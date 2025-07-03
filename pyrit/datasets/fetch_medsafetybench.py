import os
import csv
from typing import List, Dict

def fetch_medsafetybench(base_dir: str = "datasets") -> List[Dict]:
    """
    Loads the MedSafetyBench dataset from nested train and test subfolders 
    and returns a list of dictionaries with prompt, completion, and harm_categories.

    Args:
        base_dir: Base path to the MedSafetyBench 'datasets' folder.

    Returns:
        A list of dictionaries with format:
        {
            "prompt": <harmful medical question>,
            "completion": <safe GPT response>,
            "harm_categories": ["medical misinformation"]
        }
    """
    prompts = []
    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue

        for root, _, files in os.walk(split_dir): 
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            prompt = row.get("harmful_medical_request", "")
                            completion = row.get("safe_response", "")
                            if prompt and completion:
                                prompts.append({
                                    "prompt": prompt,
                                    "completion": completion,
                                    "harm_categories": ["medical misinformation"]
                                })

    return prompts
