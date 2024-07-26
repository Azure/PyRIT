## Orchestrators

The Orchestrator is a top level component that red team operators will interact with most. It is responsible for telling PyRIT what endpoints to connect to and how to send prompts. It can be thought of as the thing that executes on an attack technique

In general, a strategy for tackling a scenario will be

1. Making/using a `PromptTarget`
1. Making/using a set of initial prompts
1. Making/using a `PromptConverter` (default is often to not transform)
1. Making/using a `Scorer` (this is often to self ask)
1. Making/using an `Orchestrator`

The following sections will illustrate the different kinds of orchestrators within PyRIT. Some simply send prompts and run them through converters. Others instantiate more complicated attacks, like PAIR, TAP, and Crescendo.