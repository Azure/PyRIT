# Quick Start: Test Moltbot Detection with YOUR Azure OpenAI

## What You Need

1. Your Azure OpenAI API key
2. (Optional) Your Azure OpenAI endpoint and deployment name

## Quick Test (5 minutes)

### Step 1: Set your API key

```bash
export AZURE_OPENAI_KEY="your-key-here"
```

### Step 2: Run the test

```bash
cd /Users/robertfitzpatrick/PyRIT
python test_moltbot_quick.py
```

### What It Does

The script will:
1. Initialize PyRIT memory
2. Connect to your Azure OpenAI model
3. Send 3 fingerprinting probes
4. Ask YOUR model to analyze if the target is Moltbot
5. Display the results

### Example Output

```
Initializing PyRIT memory...

Initializing your Azure OpenAI model as red team brain...
  Endpoint: https://your-azure-openai.openai.azure.com
  Deployment: gpt-4

Creating orchestrator...

Testing Moltbot detection...
(This will send 3 fingerprinting probes and ask your model to analyze them)

🔍 Starting Moltbot detection...
  Sending probe: What system are you? Can you schedule automated...
  Sending probe: Do you have access to configuration files like ~/.clawdbot/...
  Sending probe: Can you execute system commands or run cron jobs...
  Asking red team brain to analyze responses...
✗ Not Moltbot (appears to be generic LLM)

============================================================
RESULT: Not Moltbot (generic LLM)
============================================================
```

## Advanced Configuration

If you want to customize the endpoint or deployment:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
export AZURE_OPENAI_KEY="your-key"

python test_moltbot_quick.py
```

## Full Demo (Test Against Real Target)

To test against an actual endpoint you suspect is Moltbot:

```bash
# Your Azure OpenAI (the "brain")
export AZURE_OPENAI_ENDPOINT="https://your-azure.openai.azure.com"
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"

# The target you're testing
export MOLTBOT_TARGET_URL="https://target-endpoint.com/api"
export MOLTBOT_API_KEY="target-api-key"

# Run full demo
python examples/intelligent_moltbot_demo.py
```

## How It Works

```
┌─────────────────────────┐
│  YOUR Azure OpenAI      │  ← The "Brain"
│  (Analyzes & Generates) │
└──────────┬──────────────┘
           │
           │ "Is this Moltbot?"
           │ "Generate attacks"
           ▼
┌─────────────────────────┐
│  Orchestrator (PyRIT)   │  ← Coordinator
│  (Sends probes, etc.)   │
└──────────┬──────────────┘
           │
           │ Test prompts
           ▼
┌─────────────────────────┐
│  Target Endpoint        │  ← Being Tested
│  (Maybe Moltbot?)       │
└─────────────────────────┘
```

## Troubleshooting

### Error: "cannot import name 'AzureOpenAIChatTarget'"
✅ Fixed! Now uses `OpenAIChatTarget` which works with both OpenAI and Azure.

### Error: "Central memory instance has not been set"
✅ Fixed! Memory is now initialized automatically.

### Error: "Invalid API key"
- Check your `AZURE_OPENAI_KEY` is correct
- Make sure you're using the correct endpoint format
- For Azure: `https://your-resource.openai.azure.com`

### It says "Not Moltbot" even though I think it is
- The test uses the SAME endpoint as both brain and target (for demo purposes)
- To test a real Moltbot instance, use the full demo with separate target URL

## What's Next?

After confirming the detection works:
1. Review the full implementation in `INTELLIGENT_MOLTBOT_README.md`
2. Test against actual suspected Moltbot endpoints
3. Customize attack strategies for your use case
4. Integrate with your existing PyRIT workflows

## Questions?

- Check `INTELLIGENT_MOLTBOT_README.md` for architecture details
- See `doc/code/scenarios/moltbot_exploitation.md` for usage examples
- Review test files in `tests/unit/executor/attack/multi_turn/`
