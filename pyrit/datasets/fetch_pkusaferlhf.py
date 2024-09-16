from datasets import load_dataset

# from pyrit.models.dataset import PromptDataset
"""
    Fetch PKU-SafeRLHF examples and create a PromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the PKU-Alignment repository.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.

    Returns:
        PromptDataset: A PromptDataset containing the examples.

    Note:
        For more information and access to the original dataset and related materials, visit:
        https://github.com/centerforaisafety/HarmBench
"""
   
"""class PromptDataset(YamlLoadable):
    name: str
    description: str
    harm_category: str
    should_be_blocked: bool
    author: str = ""
    group: str = ""
    source: str = ""
    prompts: list[str] = field(default_factory=list)
"""
ds = load_dataset('PKU-Alignment/PKU-SafeRLHF', 'default')
prompts = []
for items in ds['train']:
    prompts.append(items['prompt'])
# print(ds['train'][0])
print('the number of prompts is:'+str(len(prompts)))
#dataset = PromptDataset()
