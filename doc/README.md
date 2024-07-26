# Recommended Docs Reading Order

Most of our documentation is located within the `doc` directory. Of course, there is no one way to read the documentation, and it's very okay to jump around. But here is a recommended reading order.

1. [How to Guide](./how_to_guide.ipynb): This provides an overview of the PyRIT framework.
1. **Installation**: Before diving in, it's useful to have a working version so you can follow along.
  - [Setup](./setup/) includes any help setting PyRIT and related resources up as a user.
  - [Contributing](./contributing) includes information for people contributing to the project.
1. [Architecture](./code/architecture.md): This section goes over all the components at a high level. Any single component is difficult to understand without some knowledge of the others. 
1. [Orchestrators](./code/orchestrators/) are the top-most level component of PyRIT, and going through their usage can help users understand where all components fit.
1. [Targets](./code/targets/) are the endpoints that PyRIT sends prompts to. At a minimum, nearly any scenario where PyRIT is used will need targets. This section will dive into what targets are available and how to use them.
1. [Converters](./code/converters/) transform prompts from one format to another. This is one of the most powerful capabilities within PyRIT.
1. [Scorers](./code/scoring/) are how PyRIT makes decisions and records output.
1. [Memory](./code/memory/) is how PyRIT components communicate about the state of things.

Misc Extra Docs:

- [Deployment](./deployment/) includes code to download, deploy, and score open-source models (such as those from Hugging Face) on Azure.
