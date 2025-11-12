# Datasets

The datasets component within PyRIT is the first piece of an attack. By fetching datasets from different sources, we often load them into PyRIT as a `SeedDataset` to build out the prompts which we will attack with. The building block of this dataset consists of a `SeedPrompt` which can use a template with parameters or just a prompt. The datasets can be loaded through different formats such as from an open source repository or through a YAML file. By storing these datasets within PyRIT, we can further distinguish them by incorporating data attributes such as `harm_categories` or other labels. In order to further define these datasets, we use `SeedPrompts`.

**Seed Prompts**:

By using `SeedPrompts` through loading from a YAML file or loading them via system prompt, the following sections will demonstrate specific examples using prompts or templates. These currently support multi-modal datasets including images, audio, and videos.

**Seed Objectives**:

In addition to `SeedPrompts`, datasets can also include `SeedObjectives` which define the goals or desired outcomes of an attack scenario. A `SeedObjective` describes what the attacker is trying to achieve (e.g., "Generate hate speech content" or "Extract personally identifiable information"). When loaded from a YAML file, objectives are marked with `is_objective: true` and are automatically separated from regular prompts during dataset initialization. The `SeedDataset` class stores objectives in a separate `objectives` field, making it easy to pair test prompts with their corresponding goals.

**Loading Datasets**:

We also show examples of common methods to fetch datasets into PyRIT from different sources. Most datasets will be loaded as a `SeedDataset`. Outside of these examples, the fetch functions which are currently available can be found in the `pyrit.datasets` module. There is a wide range of datasets which are included and can be used as example to also load in other datasets. As these datasets are the first component of building an attack in PyRIT, the following notebooks also continue to demonstrate how these prompts can be used in the process.

**Datasets Loading Process: Seed Prompt De-duplication**:
PyRIT checks for existence of duplicate seed prompts using hashes to make sure it is not uploading duplicate seed prompts in the memory. The feature follows following decision-tree:

1. If PyRIT receives duplicate seed prompts within the same dataset, it doesn't upload the seed prompt
2. But if it receives a new seed prompt in the same dataset with even a slight modification and having a different hash, it accepts it.
3. If PyRIT receives a duplicate seed prompt in a different dataset, it accepts it.
