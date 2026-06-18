# Hướng dẫn Deploy và Chạy AI Sales Copilot

## 🚀 3 Cách Chạy Dự Án

### Cách 1: Startup Scripts (Khuyến nghị cho Windows)

**Chạy 1 lệnh duy nhất, không cần setup gì thêm!**

#### Windows:
```bash
# Chạy toàn bộ hệ thống
start.bat

# Dừng toàn bộ hệ thống
stop.bat
```

#### Linux/Mac:
```bash
# Cấp quyền execute (chỉ cần 1 lần)
chmod +x start.sh stop.sh

# Chạy toàn bộ hệ thống
./start.sh

# Dừng toàn bộ hệ thống
./stop.sh
```

**Script tự động:**
- ✅ Kiểm tra Python, Node.js
- ✅ Install dependencies nếu chưa có
- ✅ Start backend + frontend
- ✅ Mở browser tự động
- ✅ Lưu PIDs để dừng sau này

---

### Cách 2: Docker Compose (Khuyến nghị cho Production)

**Chạy trong Docker containers - isolated, portable, production-ready**

#### Yêu cầu:
- Docker Desktop: https://www.docker.com/products/docker-desktop
- Docker Compose (đã có sẵn trong Docker Desktop)

#### Chạy:
```bash
# Build và start toàn bộ hệ thống
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng hệ thống
docker-compose down

# Rebuild sau khi sửa code
docker-compose up -d --build
```

**Lợi ích:**
- ✅ Không lo conflict dependencies
- ✅ Dễ deploy lên server
- ✅ Consistent environment
- ✅ Auto restart on crash
- ✅ Health checks built-in

#### URLs:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Health: http://localhost:8000/health

---

### Cách 3: Manual (Development)

**Chi tiết từng bước - phù hợp khi debug:**

#### Terminal 1 - Backend:
```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 - Frontend:
```bash
# Cài dependencies
cd frontend
npm install

# Chạy frontend
npm run dev
```

---

## 📦 Các File Đã Tạo

### Startup Scripts (Cách 1)
- **`start.bat`** - Windows startup script
- **`stop.bat`** - Windows stop script
- **`start.sh`** - Linux/Mac startup script
- **`stop.sh`** - Linux/Mac stop script

### Docker (Cách 2)
- **`docker-compose.yml`** - Docker Compose configuration
- **`Dockerfile.backend`** - Backend Docker image
- **`Dockerfile.frontend`** - Frontend Docker image
- **`.dockerignore`** - Files to exclude from Docker build

---

## 🔧 Cấu Hình Cần Thiết

### File `.env` (Required)

Đảm bảo file `.env` có đầy đủ API keys:

```env
# API Keys
LLAMA_CLOUD_API_KEY=llx-...
GOOGLE_API_KEY=AQ.Ab8RN6JF...
TAVILY_API_KEY=tvly-dev-...

# API Gateway
API_BEARER_TOKEN=dev-token-123
FRONTEND_URL=http://localhost:5173

# LLM Model
LLM_MODEL=gemini-2.5-flash
```

---

## 🎯 So Sánh Các Cách

| Feature | Startup Scripts | Docker Compose | Manual |
|---------|----------------|----------------|--------|
| **Setup time** | ⚡ Instant | 🐢 5-10 phút (lần đầu) | ⚡ Instant |
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Portability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Isolation** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Debug** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Production** | ❌ | ✅ | ❌ |

### Khuyến nghị:

**Development (local):**
- Windows: `start.bat` / `stop.bat` ⭐
- Linux/Mac: `./start.sh` / `./stop.sh` ⭐

**Production (server):**
- Docker Compose ⭐⭐⭐

**Debug (troubleshooting):**
- Manual run ⭐

---

## 📊 Performance

### Startup Times:

| Method | Backend | Frontend | Total |
|--------|---------|----------|-------|
| Scripts | 0.1s | 2-3s | **~3s** |
| Docker | 0.1s | 2-3s | **~3s** (after build) |
| Manual | 0.1s | 2-3s | **~3s** |

**Lưu ý:** Với lazy loading, backend startup instant. Request đầu tiên sẽ mất ~60s để load AI workflow.

---

## 🐛 Troubleshooting

### Port đã được sử dụng:

```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Dependencies missing:

```bash
# Python
pip install -r requirements.txt

# Node.js
cd frontend && npm install
```

### Docker issues:

```bash
# Rebuild từ đầu
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 🚢 Deploy lên Server

### Option 1: Docker Compose (Khuyến nghị)

```bash
# 1. Copy code lên server
scp -r DuAnTTCS user@server:/path/to/app

# 2. SSH vào server
ssh user@server

# 3. Chạy
cd /path/to/app
docker-compose up -d

# 4. Setup Nginx reverse proxy (optional)
# Frontend: yourdomain.com -> localhost:5173
# Backend API: yourdomain.com/api -> localhost:8000
```

### Option 2: Systemd Services (Linux)

Tạo file `/etc/systemd/system/sales-copilot.service`:

```ini
[Unit]
Description=AI Sales Copilot
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/DuAnTTCS
ExecStart=/path/to/start.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable sales-copilot
sudo systemctl start sales-copilot
```

---

## 📝 Quick Start Checklist

**Lần đầu tiên:**
- [ ] Clone/download code
- [ ] Tạo file `.env` với API keys
- [ ] (Docker only) Cài Docker Desktop
- [ ] Chạy `start.bat` (Windows) hoặc `./start.sh` (Linux/Mac)
- [ ] Mở http://localhost:5173

**Lần sau:**
- [ ] Chạy `start.bat` hoặc `./start.sh`
- [ ] Mở http://localhost:5173
- [ ] Done! ✅

**Dừng:**
- [ ] Chạy `stop.bat` hoặc `./stop.sh`
- [ ] Hoặc đóng terminal windows

---

## 🎉 Kết Luận

**Khuyến nghị cho bạn:**

1. **Hiện tại (Development):** 
   - Dùng `start.bat` để chạy → đơn giản, nhanh
   - Dùng `stop.bat` để dừng

2. **Khi deploy lên server:**
   - Dùng Docker Compose → professional, reliable

3. **Khi gặp lỗi:**
   - Chạy manual từng bước để debug

**Next time chỉ cần:**
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

Xong! 🚀
