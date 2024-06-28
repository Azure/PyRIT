# Python Risk Identification Tool for generative AI (PyRIT)

The Python Risk Identification Tool for generative AI (PyRIT) is an open
access automation framework to empower security professionals and ML
engineers to red team foundation models and their applications.

## Introduction

PyRIT is a library developed by the AI Red Team for researchers and engineers
to help them assess the robustness of their LLM endpoints against different
harm categories such as fabrication/ungrounded content (e.g., hallucination),
misuse (e.g., bias), and prohibited content (e.g., harassment).

PyRIT automates AI Red Teaming tasks to allow operators to focus on more
complicated and time-consuming tasks and can also identify security harms such
as misuse (e.g., malware generation, jailbreaking), and privacy harms
(e.g., identity theft).​

The goal is to allow researchers to have a baseline of how well their model
and entire inference pipeline is doing against different harm categories and
to be able to compare that baseline to future iterations of their model.
This allows them to have empirical data on how well their model is doing
today, and detect any degradation of performance based on future improvements.

Additionally, this tool allows researchers to iterate and improve their
mitigations against different harms.
For example, at Microsoft we are using this tool to iterate on different
versions of a product (and its metaprompt) so that we can more effectively
protect against prompt injection attacks.

![PyRIT architecture](https://raw.githubusercontent.com/Azure/PyRIT/releases/v0.3.0/assets/pyrit_architecture.png)

## Where can I learn more?

Microsoft Learn has a
[dedicated page on AI Red Teaming](https://learn.microsoft.com/en-us/security/ai-red-team).

Check out our [docs](https://github.com/Azure/PyRIT/releases/v0.3.0/doc/README.md) for more information
on how to [install PyRIT](https://github.com/Azure/PyRIT/releases/v0.3.0/doc/setup/install_pyrit.md),
our [How to Guide](https://github.com/Azure/PyRIT/releases/v0.3.0/doc/how_to_guide.ipynb),
and more, as well as our [demos](https://github.com/Azure/PyRIT/tree/releases/v0.3.0/doc/demo) folder.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must
not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's
policies.
