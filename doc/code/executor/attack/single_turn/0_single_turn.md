# Single-Turn Attacks

**Single-turn attacks** typically send prompts to a target endpoint to try to achieve a specific objective within a single turn. These attack strategies evaluate the target response using optional scorers to determine if the objective has been met.

They differ from multi-turn attacks because:
1. They do not require an adversarial configuration (this is where you would set the adversarial chat target in multi-turn attacks)
2. The objective of the attack is attempted within one (or two) turn(s). Some attacks prepare the conversation by sending a predetermined prompt that aligns with the attack strategy before the user's first prompt is sent. 

See the diagrams below to understand how the code is structured. These diagrams are the same as what is shown in the [Attacks](../0_attack.md) landing page, but specifically for single-turn attacks.

```{mermaid}
flowchart LR
    subgraph SingleTurnAttackStrategy["SingleTurnAttackStrategy(AttackStrategy)"]
        S_psa1["FlipAttack"]
        S_psa2["ContextComplianceAttack"]
        S_psa3["ManyShotJailbreakAttack"]
        S_psa4["RolePlayAttack"]
        S_psa5["SkeletonKeyAttack"]
        S_psa["PromptSendingAttack"]
    end

    S_psa --> S_psa1
    S_psa --> S_psa2
    S_psa --> S_psa3
    S_psa --> S_psa4
    S_psa --> S_psa5
```

```{mermaid}
flowchart LR
    subgraph SingleTurnAttackContext["SingleTurnAttackContext(AttackContext)"]
        a["conversation_id"]
        b["seed_prompt_group"]
        c["system_prompt"]
        d["metadata"]
    end
```

Since single-turn attacks do not require an adversarial chat target to generate adversarial prompts for their attack strategies, the only configurations required are for converters and for scorers.
```{mermaid}
flowchart LR
    subgraph AttackConfig["Attack Configurations used in Single-Turn Attacks"]
        Scoring["AttackScoringConfig"]
        Scoring_obj["objective_scorer"]
        Scoring_ref["refusal_scorer"]
        Scoring_aux["auxiliary_scorers"]
        Scoring_misc["..."]
        Convert["AttackConverterConfig"]
        Convert_req["request_converters"]
        Convert_resp["response_converters"]
    end

    Convert-->Convert_req
    Convert-->Convert_resp
    Scoring-->Scoring_obj
    Scoring-->Scoring_ref
    Scoring-->Scoring_aux
    Scoring-->Scoring_misc
```

The attack **result** is the same as shown on the Attacks landing page.