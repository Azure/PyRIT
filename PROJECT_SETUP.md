# PyRIT UI Project - Setup Complete

A modern TypeScript + Fluent UI frontend with FastAPI backend for PyRIT.

## 📁 Project Structure

```
/workspace/
├── pyrit/backend/              # FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── README.md              # Backend documentation
│   ├── routes/                # API endpoints
│   │   ├── __init__.py
│   │   ├── chat.py           # Chat operations
│   │   ├── health.py         # Health checks
│   │   └── targets.py        # Target management
│   ├── models/               # Pydantic models
│   │   ├── __init__.py
│   │   ├── requests.py       # Request schemas
│   │   └── responses.py      # Response schemas
│   └── services/             # Business logic
│       ├── __init__.py
│       └── chat_service.py   # Chat service implementation
│
├── frontend/                   # TypeScript + React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── README.md
│   ├── .gitignore
│   └── src/
│       ├── main.tsx           # React entry point
│       ├── App.tsx            # Main app component
│       ├── types/
│       │   └── index.ts       # TypeScript types
│       ├── services/
│       │   └── api.ts         # API client
│       ├── styles/
│       │   └── global.css     # Global styles
│       └── components/
│           ├── Chat/
│           │   ├── ChatWindow.tsx
│           │   ├── MessageList.tsx
│           │   └── InputBox.tsx
│           ├── Sidebar/
│           │   └── Navigation.tsx
│           └── Layout/
│               └── MainLayout.tsx
│
├── pyproject.toml             # Updated with FastAPI dependencies
└── start-dev.sh               # Start both servers
```

## 🚀 Getting Started

### 1. Install Backend Dependencies

```bash
# Install PyRIT with FastAPI support
pip install -e ".[fastapi]"
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Run Development Servers

**Option A: Start both servers together**
```bash
./start-dev.sh
```

**Option B: Start separately**

Terminal 1 (Backend):
```bash
python -m pyrit.backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

### 4. Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **CORS** - Cross-origin requests enabled

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Fluent UI v9** - Microsoft's design system
- **Vite** - Fast build tool
- **Axios** - HTTP client

## 📋 API Endpoints

### Health & Status
- `GET /api/health` - Health check
- `GET /api/version` - Version info

### Chat Operations
- `POST /api/chat` - Send message
- `GET /api/chat/conversations` - List conversations
- `GET /api/chat/conversations/{id}` - Get conversation
- `DELETE /api/chat/conversations/{id}` - Delete conversation

### Target Management
- `GET /api/targets` - List prompt targets
- `GET /api/targets/{id}` - Get target info

## 🎨 Frontend Features

### Current Implementation
- ✅ Modern chat interface with Fluent UI
- ✅ Message history display
- ✅ Target selection sidebar
- ✅ Real-time API communication
- ✅ TypeScript type safety
- ✅ Responsive layout

### Components
- **MainLayout** - App shell with sidebar and main content
- **Navigation** - Sidebar with target selection
- **ChatWindow** - Main chat container
- **MessageList** - Message display with scrolling
- **InputBox** - Message input with keyboard shortcuts

## 🔧 Configuration

### Backend Environment Variables
- `PYRIT_API_HOST` - Default: 0.0.0.0
- `PYRIT_API_PORT` - Default: 8000
- `PYRIT_API_RELOAD` - Default: false

### Frontend Configuration
Edit `frontend/vite.config.ts` to change:
- Development port (default: 3000)
- API proxy settings
- Build options

## 📝 Next Steps

### Backend Integration
- [ ] Connect to PyRIT's memory system
- [ ] Integrate with actual prompt target registry
- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Add request validation middleware
- [ ] Connect to PyRIT orchestrators

### Frontend Enhancements
- [ ] Add conversation history view
- [ ] Implement settings panel
- [ ] Add target configuration UI
- [ ] Real-time updates (WebSocket)
- [ ] Export conversations
- [ ] Dark mode support
- [ ] Advanced message formatting (markdown, code)

### DevOps
- [ ] Add Docker configuration
- [ ] Setup CI/CD pipelines
- [ ] Add production build scripts
- [ ] Environment configuration management

## 🔍 Code Quality

The project includes:
- TypeScript for type safety
- ESLint configuration
- Proper component structure
- API error handling
- CORS configuration for development

## 📚 Documentation

- Backend API docs: http://localhost:8000/docs (when running)
- Frontend README: `frontend/README.md`
- Backend README: `pyrit/backend/README.md`

## 🤝 Integration with PyRIT

The backend is designed to integrate with existing PyRIT components:

1. **Chat Service** (`pyrit/backend/services/chat_service.py`)
   - Currently uses in-memory storage
   - Ready to integrate with PyRIT's memory system
   - TODO: Connect to PromptTarget instances

2. **Target Routes** (`pyrit/backend/routes/targets.py`)
   - Mock data currently
   - TODO: Connect to PyRIT target registry

3. **Models** (`pyrit/backend/models/`)
   - Pydantic models for API validation
   - Aligned with PyRIT's conversation structure

## 🎯 Development Workflow

1. Make changes to backend code → FastAPI auto-reloads
2. Make changes to frontend code → Vite hot-reloads
3. API calls proxied from frontend to backend
4. Type-safe communication via TypeScript types

## ✨ Key Features

- **Modern UI**: Fluent UI provides professional Microsoft design
- **Type Safety**: TypeScript + Pydantic ensure data consistency
- **Fast Development**: Vite HMR and FastAPI auto-reload
- **API Documentation**: Auto-generated Swagger/ReDoc docs
- **Clean Architecture**: Separation of concerns (routes, services, models)
- **Extensible**: Easy to add new endpoints and components

---

**Built for PyRIT - Python Risk Identification Tool for LLMs**
