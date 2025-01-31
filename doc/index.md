# PyRIT

Welcome to the Python Risk Identification Tool for generative AI (PyRIT)! PyRIT is designed to be a flexible and extensible tool that can be used to assess the security and safety issues of generative AI systems in a variety of ways.

Before starting with AI Red Teaming, we recommend reading the following article from Microsoft:
["Planning red teaming for large language models (LLMs) and their applications"](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming).

Generative AI systems introduce many categories of risk, which can be difficult to mitigate even with a red teaming
plan in place. To quote the article above, "with LLMs, both benign and adversarial usage can produce
potentially harmful outputs, which can take many forms, including harmful content such as hate speech,
incitement or glorification of violence, or sexual content." Additionally, a variety of security risks
can be introduced by the deployment of an AI system.

## Recommended Docs Reading Order

There is no single way to read the documentation, and it's perfectly fine to jump around. However, here is a recommended reading order. Note that in many sections there are numbered documentation pages. If there is no number attached, it is supplemental and the recommended reading order would be to skip it on a first pass.

1. [Cookbooks](./cookbooks/README.md): This provides an overview of PyRIT. It's useful to have an installation, but this is a good place to look to see PyRIT in action.
2. **Installation**: Before diving in, it's useful to have a working version so you can follow along.
   - [Setup](./setup/install_pyrit.md): Includes help setting up PyRIT and related resources for users.
   - [Contributing](./contributing/README.md): Contains information for people contributing to the project.
3. [Architecture](./code/architecture.md): This section provides a high-level overview of all the components. Understanding any single component is difficult without some knowledge of the others.
4. [Orchestrators](./code/orchestrators/0_orchestrator.md): These are the top-level components of PyRIT. Reviewing their usage can help users understand where all components fit.
5. [Datasets](./code/datasets/0_dataset.md): This is the first piece of building an attack using seed prompts and fetching datasets.
6. [Targets](./code/targets/0_prompt_targets.md): These are the endpoints that PyRIT sends prompts to. Nearly any scenario where PyRIT is used will need targets. This section dives into what targets are available and how to use them.
7. [Converters](./code/converters/0_converters.ipynb): These transform prompts from one format to another. This is one of the most powerful capabilities within PyRIT.
8. [Scorers](./code/scoring/0_scoring.md): These are how PyRIT makes decisions and records output.
9. [Memory](./code/memory/0_memory.md): This is how PyRIT components communicate about the state of things.
10. [Auxiliary Attacks](./code/auxiliary_attacks/0_auxiliary_attacks.ipynb): (Optional) Attacks and techniques that do not fit into the core PyRIT functionality.

Miscellaneous Extra Docs:

- [Deployment](./deployment/README.md): Includes code to download, deploy, and score open-source models (such as those from Hugging Face) on Azure.


Ongoing:

- [Blogs](./blog/README.md): Include notable new changes and are a good way to stay up to date.
