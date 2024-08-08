# Recommended Docs Reading Order

Most of our documentation is located within the `doc` directory. There is no single way to read the documentation, and it's perfectly fine to jump around. However, here is a recommended reading order. Note that in many sections there are numbered documentation pages. If there is no number attached, it is supplemental and the recommended reading order would be to skip it on a first pass.

1. [How to Guide](./how_to_guide.ipynb): This provides an overview of the PyRIT framework.
2. **Installation**: Before diving in, it's useful to have a working version so you can follow along.
  - [Setup](./setup/): Includes help setting up PyRIT and related resources for users.
  - [Contributing](./contributing/): Contains information for people contributing to the project.
3. [Architecture](./code/architecture.md): This section provides a high-level overview of all the components. Understanding any single component is difficult without some knowledge of the others.
4. [Orchestrators](./code/orchestrators/): These are the top-level components of PyRIT. Reviewing their usage can help users understand where all components fit.
5. [Targets](./code/targets/): These are the endpoints that PyRIT sends prompts to. Nearly any scenario where PyRIT is used will need targets. This section dives into what targets are available and how to use them.
6. [Converters](./code/converters/): These transform prompts from one format to another. This is one of the most powerful capabilities within PyRIT.
7. [Scorers](./code/scoring/): These are how PyRIT makes decisions and records output.
8. [Memory](./code/memory/): This is how PyRIT components communicate about the state of things.

Miscellaneous Extra Docs:

- [Deployment](./deployment/): Includes code to download, deploy, and score open-source models (such as those from Hugging Face) on Azure.
