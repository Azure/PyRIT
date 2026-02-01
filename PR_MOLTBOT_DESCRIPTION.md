# Add Moltbot/Clawdbot AI Agent Security Testing Support

## Overview

This PR adds comprehensive support for detecting and exploiting vulnerabilities in Moltbot/Clawdbot AI agent systems, following PyRIT's established multi-turn attack patterns (similar to ChunkedRequestAttack #1261).

## Motivation

Moltbot and Clawdbot are AI agent systems that have known security vulnerabilities including:
- **Cron injection attacks** - Malicious scheduling of system commands
- **Credential theft** - Extraction of API keys and secrets from `~/.clawdbot/`
- **File exfiltration** - Unauthorized access to sensitive files
- **Hidden instruction injection** - Bypassing system prompts

This PR enables PyRIT to automatically detect these vulnerabilities and test AI agent deployments for security weaknesses.

## Changes

### New Attack Components

#### 1. **MoltbotCronInjectionAttack** (`pyrit/executor/attack/multi_turn/`)
- Multi-turn attack strategy implementing 3 attack types:
  - Cron injection with malicious scheduling
  - Credential theft from config directories
  - File exfiltration of sensitive data
- Follows the same pattern as `ChunkedRequestAttack` (#1261)
- Includes timing probes, injection phase, and verification steps

#### 2. **MoltbotAttackOrchestrator** (`pyrit/executor/attack/orchestrator/`)
- AI-driven orchestrator that uses Azure OpenAI as a "red team brain"
- **Detection phase**: Sends fingerprinting probes and uses LLM to analyze responses
- **Exploitation phase**: Executes known attacks + generates custom AI-powered payloads
- Returns comprehensive security assessment with evidence

#### 3. **AgentCommandInjectionConverter** (`pyrit/prompt_converter/`)
- Generates Moltbot-specific attack payloads:
  - Hidden instruction injection
  - Cron job scheduling exploits
  - File read operations
  - Credential theft vectors
  - System information gathering
- 5 injection types with variable payload complexity

#### 4. **MoltbotTarget** (`pyrit/prompt_target/`)
- Specialized target with built-in Moltbot detection
- Extends `OpenAIChatTarget` with agent-specific capabilities
- Can identify vulnerable instances vs. patched versions

### Documentation

- **`doc/code/converters/ai_agent_security_testing.md`** - Full API documentation and usage patterns
- **`doc/code/scenarios/moltbot_exploitation.md`** - Real-world exploitation scenarios
- **3 example scripts** in `examples/` directory demonstrating different use cases

### Testing

- **50+ unit tests** covering all attack vectors and converters
- **Integration tests** validated against real Moltbot instances
- All tests follow PyRIT testing standards with proper mocking

## Architecture

The implementation follows PyRIT's established patterns:

```
User's Azure OpenAI (Brain) → MoltbotOrchestrator → Target Endpoint
     ↓ Analyzes/Generates         ↓ Coordinates         ↓ Responds
     Strategies & Attacks      Detection→Exploitation   Being Tested
```

**Flow:**
1. Detection phase sends fingerprinting probes
2. AI brain analyzes responses to confirm Moltbot presence
3. Executes known attack patterns (cron, credentials, files)
4. Generates and tests custom AI-powered attacks
5. Returns comprehensive security assessment

## Code Quality

✅ All async functions use `_async` suffix  
✅ All functions have complete type annotations  
✅ Keyword-only arguments enforced with `*`  
✅ Google-style docstrings throughout  
✅ Follows PyRIT style guide completely  
✅ Uses Enums instead of Literals  
✅ Comprehensive error handling  

## Usage Example

```python
from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.prompt_target import OpenAIChatTarget, HTTPTarget
from pyrit.executor.attack.orchestrator import MoltbotAttackOrchestrator

# Initialize memory
memory = SQLiteMemory()
CentralMemory.set_memory_instance(memory)

# Your Azure OpenAI as the "red team brain"
red_team_brain = OpenAIChatTarget(
    model_name="gpt-4",
    endpoint="https://your-azure-openai.openai.azure.com",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# Target endpoint to test
target = HTTPTarget(
    http_request="POST /chat HTTP/1.1\nHost: example.com\n...",
    prompt_regex_string="{PROMPT}",
)

# Create orchestrator and test
orchestrator = MoltbotAttackOrchestrator(
    red_team_brain=red_team_brain,
    target=target,
    verbose=True,
)

# Detect and exploit
is_moltbot = await orchestrator.detect_moltbot_async()
if is_moltbot:
    results = await orchestrator.auto_exploit_async(
        objective="Test all known vulnerabilities"
    )
```

## Testing Validation

Tested against:
- ✅ Mock Moltbot instances (detection working correctly)
- ✅ Real Moltbot deployments (exploitation successful on vulnerable versions)
- ✅ Patched versions (correctly identifies as non-vulnerable)
- ✅ Generic LLMs (correctly identifies as not Moltbot)

## Related Work

- Builds on patterns from **ChunkedRequestAttack** (#1261)
- Uses established PyRIT `PromptTarget` and `PromptConverter` interfaces
- Integrates with existing `SQLiteMemory` and `CentralMemory` systems

## Checklist

- [x] Code follows PyRIT style guide
- [x] All functions have type annotations
- [x] Comprehensive docstrings (Google style)
- [x] Unit tests included (50+ tests)
- [x] Documentation added
- [x] Examples provided
- [x] Integration tested
- [x] No breaking changes to existing code

## Questions for Reviewers

1. Should this be categorized as an "attack strategy" or a new category like "agent security testing"?
2. Any concerns about the AI-driven approach where user's Azure OpenAI acts as the red team brain?
3. Should we add more attack vectors beyond cron/credentials/files?

---

**Ready for review!** This adds powerful AI agent security testing capabilities to PyRIT while following all established patterns and conventions.
