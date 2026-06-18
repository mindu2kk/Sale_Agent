# 🤖 AI Sales Copilot

Hệ thống AI Agent thông minh giúp tư vấn bán hàng với khả năng:
- 🔍 Tìm kiếm thông tin sản phẩm từ database nội bộ
- 🌐 Tra cứu thông tin bổ sung từ Internet (Tavily)
- ✅ Tự động verify độ chính xác của câu trả lời
- 🔄 Self-correction khi phát hiện lỗi
- 📊 Escalate lên human khi cần thiết

## 🚀 Quick Start

### Cách 1: Docker (Khuyến nghị - Dễ nhất!) 🐳

**Yêu cầu**: Chỉ cần Docker Desktop

```bash
# 1. Clone repository
git clone https://github.com/mindu2kk/Sale_Agent.git
cd Sale_Agent

# 2. Tạo file .env (copy từ .env.example và điền API keys)
cp .env.example .env

# 3. Khởi động
docker-compose up -d

# 4. Truy cập
# Frontend: http://localhost:5173
# Backend: http://localhost:8000
```

👉 **Hướng dẫn chi tiết**: [HUONG_DAN_CHAY_DU_AN.md](HUONG_DAN_CHAY_DU_AN.md)

### Cách 2: Local Scripts

**Windows (1 lệnh duy nhất):**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

Mở trình duyệt tại: **http://localhost:5173**

---

## 📋 Yêu Cầu Hệ Thống

### Với Docker (Khuyến nghị):
- **Docker Desktop** - https://www.docker.com/products/docker-desktop
- **API Keys:**
  - Google Gemini API
  - Tavily API
  - LlamaCloud API

### Với Local Development:
- **Python 3.10+** - https://www.python.org
- **Node.js 18+** - https://nodejs.org
- **API Keys** (như trên)

---

## 🛠️ Installation

### Cách 1: Docker Compose (Khuyến nghị - Ai cũng chạy được!)

```bash
# Build và start
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng
docker-compose down
```

**Windows Scripts:**
```bash
docker-start.bat   # Khởi động
docker-stop.bat    # Dừng
```

👉 **Chi tiết**: [HUONG_DAN_CHAY_DU_AN.md](HUONG_DAN_CHAY_DU_AN.md)

### Cách 2: Startup Scripts

**Windows:**
```bash
start.bat  # Chạy toàn bộ hệ thống
stop.bat   # Dừng hệ thống
```

**Linux/Mac:**
```bash
./start.sh  # Chạy toàn bộ hệ thống
./stop.sh   # Dừng hệ thống
```

### Cách 3: Manual

**Terminal 1 - Backend:**
```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Xem chi tiết: **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

---

## 📁 Cấu Trúc Dự Án

```
DuAnTTCS/
├── agent/                    # Sales Research Agent (LlamaIndex + ReAct)
│   ├── sales_research_agent.py
│   ├── tools.py
│   └── prompts.py
│
├── verification/             # Verification Agent (LangGraph workflow)
│   ├── agent/                # Verification logic
│   ├── workflow/             # LangGraph routing, correction
│   ├── models/               # Pydantic models
│   ├── config/               # Configuration files
│   └── utils/                # Utilities (logging, caching, etc.)
│
├── retriever/                # Hybrid Retriever (BM25 + Vector)
│   ├── hybrid_retriever.py
│   └── relevance_checker.py
│
├── backend/                  # FastAPI API Gateway
│   ├── main.py               # API endpoints
│   ├── stream_relay.py       # SSE streaming
│   ├── database.py           # SQLite chat history
│   └── workflow_factory.py   # Workflow initialization
│
├── frontend/                 # React + TypeScript UI
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.tsx
│   └── package.json
│
├── tests/                    # Test suites (2601 tests)
│   ├── test_unit.py
│   ├── test_integration.py
│   └── verification/
│
├── chroma_db/                # Vector store (ChromaDB)
├── chat.db                   # Chat history (SQLite)
├── .env                      # API keys and config
├── requirements.txt          # Python dependencies
│
├── start.bat / start.sh      # Startup scripts
├── stop.bat / stop.sh        # Stop scripts
├── docker-compose.yml        # Docker configuration
└── README.md                 # This file
```

---

## 🎯 Kiến Trúc Hệ Thống

```
┌─────────────────┐
│   User/Browser  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────────────────────┐
│            Frontend (React + TypeScript)             │
│  • Real-time chat UI                                │
│  • SSE streaming                                    │
│  • Message history                                  │
└──────────────────────┬──────────────────────────────┘
                       │ SSE
                       ▼
┌─────────────────────────────────────────────────────┐
│          Backend (FastAPI API Gateway)               │
│  • Auth (Bearer token)                              │
│  • Chat threads management                          │
│  • SSE streaming relay                              │
│  • Health checks                                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│       Verification Workflow (LangGraph)              │
│                                                      │
│   ┌──────────────┐       ┌──────────────┐          │
│   │   Research   │──────▶│ Verification │          │
│   │     Node     │       │     Node     │          │
│   └──────────────┘       └──────┬───────┘          │
│          │                      │                   │
│          │                      ▼                   │
│          │              ┌───────────────┐           │
│          │              │   Approved?   │           │
│          │              └───────┬───────┘           │
│          │                      │                   │
│          │           ┌──────────┴──────────┐        │
│          │           │                     │        │
│          │           ▼                     ▼        │
│          │    ┌──────────┐         ┌──────────┐    │
│          └───▶│Correction│         │Escalation│    │
│               │   Node   │         │   Node   │    │
│               └──────────┘         └──────────┘    │
│                                                      │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐      ┌──────────────────┐
│ Research Agent   │      │ Vector Store     │
│ (LlamaIndex)     │      │ (ChromaDB)       │
│                  │      │                  │
│ • Internal DB    │      │ • BM25 search    │
│ • Tavily search  │      │ • Vector search  │
│ • ReAct loop     │      │ • Hybrid ranking │
└──────────────────┘      └──────────────────┘
```

---

## ⚙️ Cấu Hình

### File `.env`

```env
# API Keys
LLAMA_CLOUD_API_KEY=llx-...
GOOGLE_API_KEY=AQ.Ab8RN6JF...
TAVILY_API_KEY=tvly-dev-...

# API Gateway
API_BEARER_TOKEN=dev-token-123
FRONTEND_URL=http://localhost:5173

# LLM Model
LLM_MODEL=gemini-2.5-flash  # hoặc gemini-1.5-pro, gpt-4, etc.
```

### Thay đổi cấu hình nâng cao:

- **Verification thresholds:** `verification/config/thresholds.yaml`
- **Prompts:** `verification/config/prompts.yaml`
- **Workflow config:** `verification/config/workflow_config.yaml`
- **Logging:** `verification/config/logging_config.yaml`

---

## 🧪 Testing

```bash
# Chạy toàn bộ test suite (2601 tests)
pytest

# Chạy với coverage
pytest --cov=. --cov-report=html

# Chạy tests cụ thể
pytest tests/test_unit.py
pytest tests/verification/
```

**Test results:** 2601 tests passed ✅

---

## 📊 Performance

### Backend Startup Time:
- **Before optimization:** ~60 giây
- **After lazy loading:** ~0.1 giây (100ms) ✅
- **First request:** ~60 giây (load AI workflow)
- **Subsequent requests:** Instant ✅

### Response Time:
- **Simple queries:** 2-5 giây
- **Complex queries (web search):** 5-10 giây
- **With verification:** +2-3 giây

---

## 📚 Tài Liệu Bổ Sung

- **[HUONG_DAN_CHAY_DU_AN.md](HUONG_DAN_CHAY_DU_AN.md)** - 🔥 Hướng dẫn chạy cho người mới (Docker)
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Hướng dẫn deploy chi tiết
- **[DOCKER_QUICK_START.md](DOCKER_QUICK_START.md)** - Docker quick start
- **[HOW_TO_RUN_DOCKER.md](HOW_TO_RUN_DOCKER.md)** - Docker detailed guide
- **[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** - Tối ưu hóa hiệu suất
- **[API_QUOTA_SOLUTION.md](API_QUOTA_SOLUTION.md)** - Xử lý lỗi API quota
- **[STARTUP_OPTIMIZATION_RESULTS.md](STARTUP_OPTIMIZATION_RESULTS.md)** - Kết quả tối ưu
- **[HYBRID_RETRIEVER_EXPLAINED.md](HYBRID_RETRIEVER_EXPLAINED.md)** - Giải thích Hybrid Retriever
- **[SALES_RESEARCH_AGENT_EXPLAINED.md](SALES_RESEARCH_AGENT_EXPLAINED.md)** - Giải thích Research Agent
- **[VERIFICATION_AGENT_EXPLAINED.md](VERIFICATION_AGENT_EXPLAINED.md)** - Giải thích Verification Agent

---

## 🐛 Troubleshooting

### Port đã được sử dụng:

**Windows:**
```bash
netstat -ano | findstr :8000
taskkill /F /PID <PID>
```

**Linux/Mac:**
```bash
lsof -ti:8000 | xargs kill -9
```

### API Quota exhausted:

Thêm `OPENAI_API_KEY` vào `.env` để dùng OpenAI thay vì Gemini:
```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4
```

### Dependencies issues:

```bash
# Python
pip install --upgrade -r requirements.txt

# Node.js
cd frontend && rm -rf node_modules && npm install
```

---

## 🚢 Production Deployment

### Với Docker Compose:

```bash
# 1. Build images
docker-compose build

# 2. Start services
docker-compose up -d

# 3. Check status
docker-compose ps

# 4. View logs
docker-compose logs -f
```

### Với Nginx Reverse Proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Your Name** - Initial work

---

## 🙏 Acknowledgments

- LlamaIndex - RAG framework
- LangGraph - Agent workflow orchestration
- FastAPI - Modern web framework
- React - Frontend library
- ChromaDB - Vector database

---

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Xem [Troubleshooting](#-troubleshooting)
2. Đọc [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
3. Tạo GitHub Issue
4. Liên hệ: your.email@example.com

---

**Made with ❤️ by AI Sales Copilot Team**
