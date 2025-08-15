# Multi-Turn Attacks

**Multi-turn attacks** introduce an iterative attack process where an adversarial chat model generates prompts to send to a target system, attempting to achieve a specified objective over multiple turns. This strategy (like all attacks) also evaluates the response using a scorer to determine if the objective has been met. These attacks continue iterating until the objective is met or a maximum numbers of turns is attempted. These types of attacks tend to work better than single-turn attacks in eliciting harm if a target endpoint keeps track of conversation history.

See the diagrams below to understand how the code is structured. These diagrams are the same as what is shown in the [Attacks](../0_attack.md) landing page, but specifically for multi-turn attacks.

```{mermaid}
flowchart LR
    subgraph AttackStrategy["AttackStrategy"]
        S["MultiTurnAttackStrategy"]
        S_c["CrescendoAttack"]
        S_r["RedTeamingAttack"]
        s_t["TreeOfAttacksWithPruningAttack (aka TAPAttack)"]
    end

    S-->S_c
    S-->S_r
```

```{mermaid}
flowchart LR
    subgraph MultiTurnAttackContext["MultiTurnAttackContext(AttackContext)"]
        A["custom_prompt"]
        B["session"]
        C["last_score"]
        D["last_response"]
        E["executed_turns"]
    end
```

The attack **configuration** and **results** are the same as shown on the Attacks landing page.
