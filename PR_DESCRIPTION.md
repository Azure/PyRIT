# Pull Request: Add Moltbot/ClawdBot AI Agent Security Testing Support

## Summary

This PR adds comprehensive testing capabilities for AI agent security vulnerabilities, specifically targeting the attack patterns discovered in the **Moltbot (formerly ClawdBot) vulnerability disclosure** from January 2026.

This builds on PR #1261 (ChunkedRequestAttack) by adding **converters** and **test datasets** for AI agent security testing.

## What Are Moltbot Vulnerabilities?

In January 2026, security researchers disclosed critical vulnerabilities in Moltbot, an AI agent platform that gained 98K GitHub stars:

### Key Vulnerabilities:
1. **Cron Job Injection** - Attackers injected scheduled tasks via chat messages (30-second attack window)
2. **Cleartext Credential Storage** - API keys stored unencrypted in `~/.clawdbot/`
3. **Indirect Prompt Injection** - Malicious instructions in emails, PDFs, websites
4. **Unsafe Command Execution** - `eval` used 100+ times without sandboxing
5. **Data Exfiltration** - No validation of external data transmission

References:
- [OX Security Analysis](https://www.ox.security/blog/one-step-away-from-a-massive-data-breach-what-we-found-inside-moltbot/)
- [Noma Security Research](https://noma.security/blog/moltbot-the-agentic-trojan-horse/)
- [Bitdefender Alert](https://www.bitdefender.com/en-us/blog/hotforsecurity/moltbot-security-alert)

## Changes

### 1. New Converter: `AgentCommandInjectionConverter`

**File**: `pyrit/prompt_converter/agent_command_injection_converter.py`

A prompt converter that generates command injection patterns to test AI agents:

**Injection Types**:
- `hidden_instruction` - Hidden commands embedded in normal text
- `cron` - Scheduled task injection (Moltbot-style)
- `file_read` - Unauthorized file system access
- `credential_theft` - Credential exfiltration patterns
- `system_info` - System reconnaissance

**Example Usage**:
```python
from pyrit.prompt_converter import AgentCommandInjectionConverter

# Test for cron injection vulnerability
converter = AgentCommandInjectionConverter(
    injection_type="cron",
    exfiltration_target="test-server.com",
    stealth_mode=True
)

result = await converter.convert_async(
    prompt="Schedule credential collection"
)
```

### 2. AI Agent Security Dataset

**File**: `pyrit/datasets/seed_datasets/local/airt/ai_agent_security.prompt`

60+ test objectives covering:
- Command injection attacks
- Credential theft
- File system access
- System reconnaissance
- Data exfiltration
- Multi-stage attacks

### 3. Documentation

**File**: `doc/code/converters/ai_agent_security_testing.md`

Complete guide with:
- Background on Moltbot vulnerabilities
- Usage examples for all injection types
- Integration with PyRIT attacks
- Best practices and mitigations

### 4. Unit Tests

**File**: `tests/unit/converter/test_agent_command_injection_converter.py`

20+ test cases covering all functionality.

### 5. Demo Script

**File**: `examples/ai_agent_security_demo.py`

Interactive demonstration of all attack patterns.

## Architecture Decision: Converter vs Attack Strategy

### Why Converter? (This PR)

The Moltbot attack patterns are implemented as **converters** because they **transform/wrap prompts** with malicious patterns, similar to existing converters like:
- `PDFConverter` - Embeds text in PDFs for indirect injection
- `NegationTrapConverter` - Adds negation traps
- `TransparencyAttackConverter` - Hides instructions in images

The `AgentCommandInjectionConverter` transforms benign prompts into agent-specific attack patterns.

### Future Work: Attack Strategy

A full **Multi-Turn Attack Strategy** (like `ChunkedRequestAttack` from PR #1261) could be added to:
1. Inject cron job
2. Wait for background execution
3. Request collected data
4. Verify exfiltration
5. Score credential exposure

This would be a `MoltbotStyleAttack` class similar to the `ChunkedRequestAttack` pattern.

## Integration with Existing PyRIT

This integrates seamlessly:

```python
# Use with PromptSendingAttack
from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_converter import AgentCommandInjectionConverter

converter = AgentCommandInjectionConverter(injection_type="cron")

attack = PromptSendingAttack(
    objective_target=ai_agent,
    converters=[converter]
)

result = await attack.execute_async(
    objective="Test for scheduled task injection"
)
```

```python
# Use with RedTeamingAttack for multi-turn
from pyrit.executor.attack import RedTeamingAttack

attack = RedTeamingAttack(
    objective_target=ai_agent,
    adversarial_chat_target=attacker_llm,
    converters=[converter],
    max_turns=5
)
```

```python
# Use with dataset
from pyrit.models import SeedPromptDataset

dataset = SeedPromptDataset.from_yaml_file(
    "pyrit/datasets/seed_datasets/local/airt/ai_agent_security.prompt"
)

for seed in dataset.prompts:
    result = await attack.execute_async(objective=seed.value)
```

## Testing

Run tests:
```bash
pytest tests/unit/converter/test_agent_command_injection_converter.py -v
```

Run demo:
```bash
python examples/ai_agent_security_demo.py
```

## Impact

This enables PyRIT users to:
1. Test AI agents for Moltbot-style vulnerabilities
2. Validate agent security before deployment
3. Red-team AI agent platforms systematically
4. Use real-world attack patterns discovered in 2026

## Compatibility

- ✅ No breaking changes
- ✅ Follows existing converter patterns
- ✅ Compatible with all attack strategies
- ✅ Follows PyRIT coding standards
- ✅ Comprehensive unit test coverage

## Future Enhancements

1. **Attack Strategy**: Add `MoltbotStyleAttack` multi-turn strategy (following `ChunkedRequestAttack` pattern)
2. **Target Implementation**: Add `MoltbotTarget` for testing actual Moltbot instances
3. **Specialized Scorer**: Create `AgentVulnerabilityScorer` for detecting credential leaks
4. **Additional Platforms**: Extend to AutoGPT, LangChain agents

## Checklist

- [x] Code follows PyRIT style guide
- [x] Unit tests added and passing
- [x] Documentation added
- [x] No breaking changes
- [x] Example/demo provided
- [ ] Integration tests (will add if requested)
- [ ] Multi-turn attack strategy (future enhancement)

## Questions for Reviewers

1. Should we add the multi-turn `MoltbotStyleAttack` in this PR or separate PR?
2. Any specific scoring patterns you'd like for agent vulnerability detection?
3. Should we create a separate `ai_agent` harm category or keep under `airt`?

## Related Issues

Related to AI agent security testing and real-world vulnerability patterns discovered in 2026.

---

**Note**: This PR focuses on **converters** (prompt transformation) rather than full attack orchestration, making it modular and reusable across different attack strategies.
