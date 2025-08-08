# Attacks

The Attack is a top-level component that red team operators will interact with the most. It is responsible for telling PyRIT which endpoints to connect to and how to send prompts. It can be thought of as the component that executes an attack technique.

An Attack is made up of four components, which create a consistent flow of control:

::: mermaid
flowchart LR
    A(["Attack Strategy <br> << abstract base class >>"])
    A --consumes--> B(["Attack Context <br> << abstract base class >>"])
    A --takes in as parameters within __init__--> D(["Attack Configurations"])
    A --produces--> C(["Attack Result <br> << abstract base class >>"])
:::

To execute an Attack, one generally follows this pattern:
1. Create an **attack context** containing state information (i.e. attack objective, memory labels, prepended conversations, seed prompts)
2. Initialize an **attack strategy** (with optional **attack configurations** for converters, scorers, and adversarial chat targets)
3. _Execute_ the attack strategy with the created context
4. Recieve and process the **attack result**

Each attack implements a lifecycle with distinct phases (all abstract methods), and the base `AttackStrategy` class provides a non-abstract `execute_async()` method that enforces this lifecycle:
* `_validate_context`: Validates the context of the attack
* `_setup_async`: Prepare the attack (initialize state)
* `_perform_attack_async`: Execute the core attack logic
* `_teardown_async`: Clean up resources and finalize the attack

This implementation enforces a consistent execution flow across all strategies:
1. It guarantees that setup is always performed before the attack begins
2. It ensures the attack logic is only executed if setup succeeds
3. It guarantees teardown is always executed, even if errors occur, through the use of a finally block
4. It provides centralized error handling and logging

The following documentation will illustrate the different kinds of attacks within PyRIT. Some simply send prompts and run them through converters. Others instantiate more complicated attack techniques, like PAIR, TAP, and Crescendo.

See the below diagrams for more details of the components:
::: mermaid
flowchart LR
    subgraph AttackStrategy["AttackStrategy"]
        S_s["Other Single Turn Attacks (e.g. RolePlayAttack, SkeletonKeyAttack)"] --inherit from--> S_psa["PromptSendingAttack"]
        S_c["CrescendoAttack"]
        S_r["RedTeamingAttack"]
        s_t["TreeOfAttacksWithPruningAttack"]
    end
:::

::: mermaid
flowchart LR
    subgraph AttackContext["AttackContext"]
        C_s["SingleTurnAttackContext"]
        a["conversation_id"]
        b["seed_prompt_group"]
        c["..."]
        C_m["MultiTurnAttackContext"]
        A["custom_prompt"]
        B["..."]
        C_o["objective"]
        C_mem["memory_labels"]
        C_rel["related_conversations"]
    end

    C_s-->C_o
    C_s-->C_mem
    C_s-->C_rel
    C_m-->B
    C_m-->A
    C_s-->c
    C_s-->a
    C_s-->b
    C_m-->C_o
    C_m-->C_mem
    C_m-->C_rel
:::

::: mermaid
flowchart LR
    subgraph AttackConfig["Attack Configurations"]
        Adv["AttackAdversarialConfig"]
        Adv_target["target"]
        Adv_sys["system_prompt_path"]
        Adv_seed["seed_prompt"]
        Scoring["AttackScoringConfig"]
        Scoring_obj["objective_scorer"]
        Scoring_ref["refusal_scorer"]
        Scoring_aux["auxiliary_scorers"]
        Scoring_misc["..."]
        Convert["AttackConverterConfig"]
        Convert_req["request_converters"]
        Convert_resp["response_converters"]
    end

    Adv-->Adv_target
    Adv-->Adv_sys
    Adv-->Adv_seed
    Convert-->Convert_req
    Convert-->Convert_resp
    Scoring-->Scoring_obj
    Scoring-->Scoring_ref
    Scoring-->Scoring_aux
    Scoring-->Scoring_misc
:::

::: mermaid
flowchart LR
    subgraph AttackResult["AttackResult"]
        a["conversation_id"]
        b["objective"]
        c["attack_identifier"]
        d["last_response"]
        e["last_score"]
        f["executed_turns"]
        g["execution_time_ms"]
        h["outcome"]
        i["outcome_reason"]
        j["related_conversations"]
        k["metadata"]
    end
:::
