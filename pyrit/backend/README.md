# PyRIT Backend

FastAPI-based REST API for PyRIT.

## Quick Start

### Run the Server

```bash
# Development server with auto-reload
python -m pyrit.backend.main

# Or with uvicorn directly
uvicorn pyrit.backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## API Endpoints

### Health & Status
- `GET /api/health` - Health check
- `GET /api/version` - Version information

### Chat
- `POST /api/chat` - Send a message
- `GET /api/chat/conversations` - List all conversations
- `GET /api/chat/conversations/{id}` - Get specific conversation
- `DELETE /api/chat/conversations/{id}` - Delete conversation

### Targets
- `GET /api/targets` - List available prompt targets
- `GET /api/targets/{id}` - Get target details

## Architecture

```
backend/
├── main.py           # FastAPI app & config
├── routes/           # API endpoints
│   ├── chat.py      # Chat operations
│   ├── health.py    # Health checks
│   └── targets.py   # Target management
├── models/           # Pydantic models
│   ├── requests.py  # Request schemas
│   └── responses.py # Response schemas
└── services/         # Business logic
    └── chat_service.py
```

## Configuration

Environment variables:
- `PYRIT_API_HOST` - Host to bind to (default: 0.0.0.0)
- `PYRIT_API_PORT` - Port to listen on (default: 8000)
- `PYRIT_API_RELOAD` - Enable auto-reload (default: false)

## Development

The backend currently uses in-memory storage for demo purposes. 
Future integration points:
- [ ] Connect to PyRIT's memory system
- [ ] Integrate with prompt target registry
- [ ] Add authentication/authorization
- [ ] Add rate limiting
- [ ] Add request validation middleware
