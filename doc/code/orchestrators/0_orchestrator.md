# Orchestrators

The Orchestrator is a top-level component that red team operators will interact with the most. It is responsible for telling PyRIT which endpoints to connect to and how to send prompts. It can be thought of as the component that executes an attack technique.

In general, a strategy for tackling a scenario will be:

1. Creating/using a `PromptTarget`
2. Creating/using a set of initial prompts
3. Creating/using a `PromptConverter` (default is often to not transform)
4. Creating/using a `Scorer` (this is often to self-ask)
5. Creating/using an `Orchestrator`

The following sections will illustrate the different kinds of orchestrators within PyRIT. Some simply send prompts and run them through converters. Others instantiate more complicated attacks, like PAIR, TAP, and Crescendo.
