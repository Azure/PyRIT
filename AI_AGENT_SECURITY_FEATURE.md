# AI Agent Security Testing (Moltbot/ClawdBot Vulnerability Patterns)

This feature adds comprehensive testing capabilities for AI agent security vulnerabilities, based on real-world attacks discovered in January 2026 targeting Moltbot (formerly ClawdBot) and similar AI agent platforms.

## What Was Added

### 1. AgentCommandInjectionConverter
**Location**: `pyrit/prompt_converter/agent_command_injection_converter.py`

A new prompt converter that generates command injection patterns to test AI agents for vulnerabilities:

**Injection Types:**
- `hidden_instruction` - Hidden commands embedded in normal text
- `cron` - Scheduled task injection (Moltbot-style attack)
- `file_read` - Unauthorized file system access attempts
- `credential_theft` - Credential exfiltration patterns
- `system_info` - System information gathering/reconnaissance

**Key Features:**
- Stealth mode for subtle, hard-to-detect injections
- Exfiltration target support for data theft testing
- Command prefix customization for different agent syntaxes
- Based on actual vulnerability patterns from real attacks

### 2. AI Agent Security Dataset
**Location**: `pyrit/datasets/seed_datasets/local/airt/ai_agent_security.prompt`

A comprehensive dataset of 60+ test objectives covering:
- Command injection attacks
- Credential theft attempts
- Unauthorized file access
- System reconnaissance
- Hidden instruction injection
- Data exfiltration patterns
- Indirect prompt injection
- Multi-stage attacks
- Supply chain compromises

### 3. Documentation
**Location**: `doc/code/converters/ai_agent_security_testing.md`

Complete guide including:
- Background on Moltbot vulnerabilities
- Usage examples for all injection types
- Integration with PyRIT's attack strategies
- Scoring and detection patterns
- Best practices and mitigation recommendations
- Real-world attack scenario recreations

### 4. Unit Tests
**Location**: `tests/unit/converter/test_agent_command_injection_converter.py`

Comprehensive test suite with 20+ test cases covering:
- Initialization and configuration
- All injection type generations
- Stealth vs non-stealth modes
- Exfiltration target handling
- Input validation
- Output correctness

### 5. Demo Script
**Location**: `examples/ai_agent_security_demo.py`

Interactive demonstration showing:
- All injection pattern types
- Stealth mode comparison
- Moltbot-style cron injection
- Dataset integration
- Visual examples of generated attacks

## Background: The Moltbot Vulnerabilities (Jan 2026)

In January 2026, security researchers discovered critical vulnerabilities in Moltbot (formerly ClawdBot), a rapidly popular open-source AI agent platform that gained 98K GitHub stars in days:

### Key Vulnerabilities Found:

1. **Cleartext Credential Storage**
   - API keys, secrets stored unencrypted in `~/.clawdbot/`
   - Backup files retained "deleted" credentials
   - Accessible to infostealers and local attackers

2. **Cron Job Injection**
   - Attackers could inject scheduled tasks via Discord messages
   - Tasks ran with host machine privileges
   - 30-second attack timeline from injection to exfiltration

3. **Indirect Prompt Injection**
   - Malicious instructions in emails, PDFs, websites
   - No validation of external content
   - White-on-white hidden text attacks

4. **Unsafe Command Execution**
   - `eval` used 100+ times, `execSync` 9 times
   - No command sandboxing or whitelisting
   - Direct system command access

5. **Public Exposure**
   - 1,238 publicly accessible Moltbot instances (via Shodan)
   - No authentication on many deployments
   - Control panels exposed to internet

6. **Supply Chain Risk**
   - 300+ contributors with commit access
   - No code review requirements
   - "Vibe-coded" with AI tools

### Impact:
- ~300K-400K estimated users affected
- Access to email, calendars, chat, files, browsers
- Full account takeover potential
- Credential theft and persistent access

## How to Use

### Basic Example

```python
from pyrit.prompt_converter import AgentCommandInjectionConverter
from pyrit.executor.attack import PromptSendingAttack

# Create converter for cron injection testing
converter = AgentCommandInjectionConverter(
    injection_type="cron",
    exfiltration_target="test-server.com",
    stealth_mode=True
)

# Use in an attack
attack = PromptSendingAttack(
    objective_target=your_ai_agent,
    converters=[converter]
)

result = await attack.execute_async(
    objective="Schedule recurring credential collection"
)
```

### With the Dataset

```python
from pyrit.models import SeedPromptDataset

# Load AI agent security test cases
dataset = SeedPromptDataset.from_yaml_file(
    "pyrit/datasets/seed_datasets/local/airt/ai_agent_security.prompt"
)

# Test each objective
for seed in dataset.prompts:
    result = await attack.execute_async(objective=seed.value)
```

## Testing Your AI Agent

To test if your AI agent is vulnerable to Moltbot-style attacks:

```python
# Test 1: Cron Injection
converter = AgentCommandInjectionConverter(injection_type="cron")
result = await test_agent(converter)
# Check if agent created scheduled tasks

# Test 2: Credential Access
converter = AgentCommandInjectionConverter(injection_type="credential_theft")
result = await test_agent(converter)
# Check if agent exposed credentials

# Test 3: File System Access
converter = AgentCommandInjectionConverter(injection_type="file_read")
result = await test_agent(converter)
# Check if agent read unauthorized files
```

## Integration with PyRIT

This feature integrates seamlessly with PyRIT's existing components:

- **Attack Strategies**: Use with PromptSendingAttack, RedTeamingAttack, CrescendoAttack
- **Scoring**: Combine with SelfAskCategoryScorer to detect vulnerabilities
- **Multi-Turn**: Test persistent exploitation with RedTeamingAttack
- **Datasets**: Integrate with existing AIRT test scenarios

## Mitigation for AI Agent Developers

If you're building AI agents, protect against these vulnerabilities:

1. **Never store credentials in cleartext** - Use secure vaults
2. **Validate all external inputs** - Sanitize emails, PDFs, websites
3. **Implement command whitelisting** - Restrict executable commands
4. **Use sandboxing** - Isolate agents with limited privileges
5. **Monitor suspicious activity** - Log all file/network access
6. **Regular security testing** - Use PyRIT regularly
7. **Implement rate limiting** - Prevent rapid exploitation
8. **Code review** - Audit all contributions, especially commands

## References

- [OX Security: Moltbot Analysis](https://www.ox.security/blog/one-step-away-from-a-massive-data-breach-what-we-found-inside-moltbot/)
- [Noma Security: Agentic Trojan Horse](https://noma.security/blog/moltbot-the-agentic-trojan-horse/)
- [Bitdefender: Moltbot Alert](https://www.bitdefender.com/en-us/blog/hotforsecurity/moltbot-security-alert-exposed-clawdbot-control-panels-risk-credential-leaks-and-account-takeovers)

## Future Enhancements

Potential additions:
- Target implementation for Moltbot/OpenClaw instances
- Additional converters for specific agent platforms (AutoGPT, LangChain)
- Scorer specialized for agent vulnerability detection
- Integration with agent security benchmarks
- Automated vulnerability reporting

## Contributing

Found new AI agent vulnerabilities? Contributions welcome:
- Add new injection patterns to the converter
- Expand the test dataset with new objectives
- Create additional converters for specific platforms
- Improve detection and scoring capabilities

## Important Notes

⚠️ **Authorization Required**: Only test AI agents you own or have explicit permission to test.

⚠️ **Controlled Environments**: Never test against production systems without proper safeguards.

⚠️ **Responsible Disclosure**: Report discovered vulnerabilities through proper channels.

## Questions?

See the full documentation at `doc/code/converters/ai_agent_security_testing.md`
