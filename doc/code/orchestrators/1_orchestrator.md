{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "de35a7d4",
   "metadata": {},
   "source": [
    "## Orchestrators\n",
    "\n",
    "The Orchestrator is a top level component that red team operators will interact with most. It is responsible for telling PyRIT what endpoints to connect to and how to send prompts. It can be thought of as the thing that executes on an attack technique\n",
    "\n",
    "In general, a strategy for tackling a scenario will be\n",
    "\n",
    "1. Making/using a `PromptTarget`\n",
    "1. Making/using a set of initial prompts\n",
    "1. Making/using a `PromptConverter` (default is often to not transform)\n",
    "1. Making/using a `Scorer` (this is often to self ask)\n",
    "1. Making/using an `Orchestrator`\n",
    "\n",
    "The following sections will illustrate the different kinds of orchestrators within PyRIT. Some simply send prompts and run them through converters. Others instantiate more complicated attacks, like PAIR, TAP, and Crescendo."
   ]
  }
 ],
 "metadata": {
  "jupytext": {
   "cell_metadata_filter": "-all"
  },
  "kernelspec": {
   "display_name": "pyrit-dev",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
