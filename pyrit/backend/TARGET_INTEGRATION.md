# PyRIT Backend - Target Integration

## Overview

The backend now integrates with actual PyRIT targets using environment variables from your `.env` file.

## How It Works

### 1. **Target Discovery**
The `TargetRegistry` service scans your environment variables and discovers configured targets:

- `OPENAI_CHAT_ENDPOINT` / `OPENAI_CHAT_KEY` / `OPENAI_CHAT_MODEL`
- `AZURE_OPENAI_GPT4O_ENDPOINT` / `AZURE_OPENAI_GPT4O_KEY`
- `AZURE_OPENAI_GPT3_5_CHAT_ENDPOINT` / `AZURE_OPENAI_GPT3_5_CHAT_KEY`
- `AZURE_ML_MANAGED_ENDPOINT` / `AZURE_ML_KEY`
- And more...

### 2. **API Endpoints**

**GET `/api/targets`** - Lists all configured targets from your `.env`:
```json
[
  {
    "id": "OpenAIChatTarget",
    "name": "OpenAI Chat",
    "type": "OpenAIChatTarget",
    "description": "Standard OpenAI Chat Completions endpoint",
    "status": "available"
  },
  {
    "id": "AzureOpenAIGPT4o",
    "name": "Azure OpenAI GPT 4o",
    "type": "OpenAIChatTarget",
    "description": "Azure OpenAI GPT-4o deployment",
    "status": "available"
  }
]
```

**POST `/api/chat`** - Send messages using a configured target:
```json
{
  "message": "Hello, test my AI system",
  "target_id": "AzureOpenAIGPT4o",
  "conversation_id": "optional-conv-id"
}
```

### 3. **Target Usage**

#### **Chat Interface**
When you select a target in the UI and send a message:
1. Frontend sends `POST /api/chat` with `target_id`
2. Backend creates target instance using `TargetRegistry`
3. Message is sent to the actual PyRIT target
4. Real response is returned to frontend

#### **Attack Operations**
For converters, scorers, and adversarial chat (future implementation):
- Uses `OpenAIChatTarget` with default `OPENAI_CHAT_*` environment variables
- Call `TargetRegistry.get_default_attack_target()` to get the instance

### 4. **Configuration**

Make sure your `.env` file has the targets you want to use:

```bash
# Standard OpenAI (used for attacks by default)
OPENAI_CHAT_ENDPOINT="https://api.openai.com/v1/chat/completions"
OPENAI_CHAT_KEY="sk-xxxxx"
OPENAI_CHAT_MODEL="gpt-4o"

# Azure OpenAI GPT-4o
AZURE_OPENAI_GPT4O_ENDPOINT="https://xxxx.openai.azure.com/openai/deployments/xxxxx/chat/completions?api-version=2024-10-21"
AZURE_OPENAI_GPT4O_KEY="xxxxx"

# Azure ML
AZURE_ML_MANAGED_ENDPOINT="https://xxxxx.westus3.inference.ml.azure.com/score"
AZURE_ML_KEY="xxxxx"
```

### 5. **Target Registry Details**

The `TargetRegistry` class provides:

```python
# Get all available targets from environment
targets = TargetRegistry.get_available_targets()

# Create a specific target instance
target = TargetRegistry.create_target_instance("AzureOpenAIGPT4o")

# Get default target for attacks/converters/scorers
attack_target = TargetRegistry.get_default_attack_target()
```

### 6. **Future Enhancements**

- [ ] Add endpoints for converters (e.g., `POST /api/convert`)
- [ ] Add endpoints for scorers (e.g., `POST /api/score`)
- [ ] Add endpoints for adversarial chat (e.g., `POST /api/adversarial-chat`)
- [ ] Support target parameter overrides from UI
- [ ] Add target health checks
- [ ] Cache target instances for performance
- [ ] Support more target types (DALL-E, TTS, Sora, etc.)

## Architecture

```
User Message
    ↓
Frontend (React/TypeScript)
    ↓
POST /api/chat
    ↓
ChatService
    ↓
TargetRegistry
    ↓
PyRIT Target Instance (OpenAIChatTarget, AzureMLChatTarget, etc.)
    ↓
Actual LLM Endpoint
    ↓
Response back through the chain
```

## Error Handling

- If no targets are configured: Returns status "not_configured"
- If API key is missing: Returns status "needs_api_key"
- If target fails: Returns error message in chat response

## Testing

```bash
# List available targets
curl http://localhost:8000/api/targets

# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello",
    "target_id": "OpenAIChatTarget"
  }'
```
