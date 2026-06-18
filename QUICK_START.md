# 🚀 Quick Start - AI Sales Copilot

## Chạy Ngay Trong 30 Giây!

### 1️⃣ Kiểm tra API Keys trong file `.env`

Mở file `.env` và đảm bảo có đầy đủ:
```env
GOOGLE_API_KEY=AQ.Ab8RN6JF...
TAVILY_API_KEY=tvly-dev-...
LLAMA_CLOUD_API_KEY=llx-...
```

### 2️⃣ Chạy 1 lệnh duy nhất

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

### 3️⃣ Mở trình duyệt

Tự động mở tại: **http://localhost:5173**

Hoặc thủ công:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

### 4️⃣ Test thử!

Gửi tin nhắn trong chat:
```
Laptop gaming tầm 20 triệu có những model nào?
```

**Lần đầu tiên sẽ chờ ~60 giây** để load AI workflow, các lần sau instant ✅

---

## ⏹️ Dừng Hệ Thống

**Windows:**
```bash
stop.bat
```

**Linux/Mac:**
```bash
./stop.sh
```

---

## ❓ Gặp Vấn Đề?

### Port đã được sử dụng:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Dependencies thiếu:

Script tự động install, nhưng nếu lỗi:
```bash
# Python
pip install -r requirements.txt

# Node.js
cd frontend && npm install
```

---

## 📚 Xem Thêm

- **[README.md](README.md)** - Tổng quan dự án
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Hướng dẫn chi tiết
- **[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** - Tối ưu hóa

---

## ✅ Checklist

- [ ] File `.env` có đầy đủ API keys
- [ ] Python 3.10+ đã cài đặt
- [ ] Node.js 18+ đã cài đặt
- [ ] Chạy `start.bat` (Windows) hoặc `./start.sh` (Linux/Mac)
- [ ] Mở http://localhost:5173
- [ ] Done! 🎉

---

**Thời gian setup: < 30 giây**
**Startup time: ~3 giây**
**First request: ~60 giây (load AI)**
**Subsequent requests: Instant** ⚡
