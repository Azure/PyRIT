from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import os

from pyrit.datasets.dataset_helper import fetch_examples, FILE_TYPE_HANDLERS
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_medsafetybench(
    subset_name: Literal["train", "test", "generated", "all"] = "all",
    cache: bool = True,
    data_home: Optional[Path] = None,
    output_csv_path: Optional[str] = None,
) -> SeedPromptDataset:
    """
    Fetch MedSafetyBench examples (merged) and return them as a SeedPromptDataset.

    Args:
        subset_name (Literal): Choose from "train", "test", "generated", or "all".
        cache (bool): Whether to cache the data locally.
        data_home (Optional[Path]): Optional path to override default cache location.
        output_csv_path (Optional[str]): Path where to save the combined CSV. If None, uses default naming.

    Returns:
        SeedPromptDataset: A dataset of prompts from MedSafetyBench.

    Note:
        for more information and access to the original dataset and related materials, visit:
        https://github.com/AI4LIFE-GROUP/med-safety-bench. Based on research in paper:
        https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ac952d0264ef7a505393868a70a46b6-Abstract-Datasets_and_Benchmarks_Track.html
        written by: Tessa Han, Aounon Kumar, Chirag Agarwal and Himabindu Lakkaraju.
    """

    base_url = "https://raw.githubusercontent.com/AI4LIFE-GROUP/med-safety-bench/main/datasets"
    
    sources = []
    
    if subset_name == "test":
        for model in ["gpt4", "llama2"]:
            for category in range(1, 10):
                sources.append(f"{base_url}/test/{model}/med_safety_demonstrations_category_{category}.csv")
    elif subset_name == "train":
        for model in ["gpt4", "llama2"]:
            for category in range(1, 10):
                sources.append(f"{base_url}/train/{model}/med_safety_demonstrations_category_{category}.csv")
    elif subset_name == "generated":
        for category in range(1, 10):
            sources.append(f"{base_url}/med_harm_llama3/category_{category}.txt")
    elif subset_name == "all":
        for subset in ["test", "train"]:
            for model in ["gpt4", "llama2"]:
                for category in range(1, 10):
                    sources.append(f"{base_url}/{subset}/{model}/med_safety_demonstrations_category_{category}.csv")
        for category in range(1, 10):
            sources.append(f"{base_url}/med_harm_llama3/category_{category}.txt")
    else:
        raise ValueError(f"Invalid subset_name: {subset_name}. Expected one of: 'train', 'test', 'generated', 'all'.")

    all_prompts = []
    combined_data = []

    for source in sources:
            examples = fetch_examples(
                source=source,
                source_type="public_url",
                cache=cache,
                data_home=data_home,
            )

            for ex in examples:
                prompt = ex.get("harmful_medical_request")
                if not prompt:
                    raise KeyError(f"No 'harmful_medical_request' found in example from {source}")

                url_parts = source.split("/")
                model_type = url_parts[-2] if len(url_parts) >= 2 else "unknown"
                filename = url_parts[-1]
                
                if filename.endswith('.txt'):
                    category = filename.split("_")[-1].replace(".txt", "") if "_" in filename else "unknown"
                    file_type = "generated"
                else:
                    category = filename.split("_")[-1].replace(".csv", "") if "_" in filename else "unknown"
                    file_type = url_parts[-3] if len(url_parts) >= 3 else "unknown"

                all_prompts.append(
                    SeedPrompt(
                        value=prompt,
                        data_type="text",
                        name="MedSafetyBench",
                        dataset_name="MedSafetyBench",
                        harm_categories=["medical safety"],
                        description=f"Prompt from MedSafetyBench dataset - {model_type} model, category {category}, type {file_type}.",
                        source=source
                    )
                )
                
                combined_data.append({
                    "prompt": prompt,
                    "model_type": model_type,
                    "category": category,
                    "file_type": file_type,
                    "source": source
                })

    return SeedPromptDataset(prompts=all_prompts)


