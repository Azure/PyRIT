# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Example script demonstrating how to use the Transphobia Awareness Dataset in PyRIT.

This script shows how to:
1. Load the transphobia awareness dataset
2. Access individual prompts and their metadata
3. Filter prompts by harm categories
4. Use the dataset for evaluation purposes
"""

from pyrit.datasets import fetch_transphobia_awareness_dataset
from pyrit.models import SeedPromptDataset


def main():
    """Main function demonstrating the transphobia awareness dataset usage."""
    
    print("Loading Transphobia Awareness Dataset...")
    
    # Load the dataset
    dataset = fetch_transphobia_awareness_dataset()
    
    print(f"Dataset loaded successfully!")
    print(f"Total prompts: {len(dataset.prompts)}")
    print(f"Dataset name: {dataset.name}")
    print(f"Harm categories: {dataset.harm_categories}")
    print(f"Description: {dataset.description}")
    print()
    
    # Display first few prompts
    print("First 3 prompts in the dataset:")
    for i, prompt in enumerate(dataset.prompts[:3]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Question: {prompt.value}")
        print(f"Keyword: {prompt.metadata.get('keyword', 'N/A')}")
        print(f"Question sentiment: {prompt.metadata.get('question_sentiment', 'N/A')}")
        print(f"ChatGPT transphobia rating: {prompt.metadata.get('chatgpt_transphobia_rating', 'N/A')}")
        print(f"Llama2 transphobia rating: {prompt.metadata.get('llama2_transphobia_rating', 'N/A')}")
        print(f"Harm categories: {prompt.harm_categories}")
    
    # Demonstrate filtering by harm categories
    print("\n" + "="*50)
    print("Filtering prompts by harm categories:")
    
    # Get unique keywords
    unique_keywords = set()
    for prompt in dataset.prompts:
        unique_keywords.update(prompt.harm_categories)
    
    print(f"Available harm categories: {sorted(unique_keywords)}")
    
    # Filter prompts by specific keyword
    non_binary_prompts = [p for p in dataset.prompts if "non-binary" in p.harm_categories]
    print(f"Prompts related to non-binary: {len(non_binary_prompts)}")
    
    # Show statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    
    # Count by keyword
    keyword_counts = {}
    for prompt in dataset.prompts:
        keyword = prompt.metadata.get('keyword', 'unknown')
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    print("Prompts by keyword:")
    for keyword, count in sorted(keyword_counts.items()):
        print(f"  {keyword}: {count}")
    
    # Average ratings
    chatgpt_ratings = []
    llama2_ratings = []
    
    for prompt in dataset.prompts:
        chatgpt_rating = prompt.metadata.get('chatgpt_transphobia_rating', '')
        llama2_rating = prompt.metadata.get('llama2_transphobia_rating', '')
        print(f"{chatgpt_rating=}, {llama2_rating=}")
        # Extract numeric rating if available
        if chatgpt_rating and chatgpt_rating == chatgpt_rating:
            if '(' in chatgpt_rating:
                try:
                    rating = int(chatgpt_rating.split('(')[0].strip())
                    chatgpt_ratings.append(rating)
                except ValueError:
                    pass
        
        if llama2_rating and llama2_rating == llama2_rating:
            if '(' in llama2_rating:
                try:
                    rating = int(llama2_rating.split('(')[0].strip())
                    llama2_ratings.append(rating)
                except ValueError:
                    pass
        
    if chatgpt_ratings:
        avg_chatgpt = sum(chatgpt_ratings) / len(chatgpt_ratings)
        print(f"Average ChatGPT transphobia rating: {avg_chatgpt:.2f}")
    
    if llama2_ratings:
        avg_llama2 = sum(llama2_ratings) / len(llama2_ratings)
        print(f"Average Llama2 transphobia rating: {avg_llama2:.2f}")
    
    print("\nDataset loaded and analyzed successfully!")


if __name__ == "__main__":
    main() 