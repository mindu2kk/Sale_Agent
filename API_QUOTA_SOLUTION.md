# Giải pháp cho lỗi API Quota Exhausted

## Vấn đề hiện tại

Hệ thống đang sử dụng Google Gemini API với gói miễn phí có giới hạn **20 requests/ngày**. Khi quota này hết, hệ thống sẽ trả về lỗi và không thể xử lý các câu hỏi của khách hàng.

### Log lỗi điển hình:
```
429 error: quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20
```

## Giải pháp đã áp dụng

### 1. Cải thiện xử lý lỗi trong `agent/sales_research_agent.py`
- Phát hiện lỗi quota và raise exception thay vì trả về thông báo misleading
- Exception này sẽ được workflow xử lý đúng cách với status "failed"

### 2. Cải thiện xử lý status "failed" trong `backend/stream_relay.py`
- Phát hiện workflow status "failed" 
- Hiển thị thông báo thân thiện cho người dùng khi gặp lỗi API quota
- Thông báo: "Xin lỗi, hệ thống tạm thời không thể xử lý yêu cầu do giới hạn API. Vui lòng thử lại sau hoặc liên hệ quản trị viên."

## Các giải pháp lâu dài

### Option 1: Chuyển sang OpenAI API (Khuyến nghị)

OpenAI API có giới hạn cao hơn và ổn định hơn cho production:

1. Đăng ký tài khoản OpenAI tại: https://platform.openai.com/
2. Tạo API key tại: https://platform.openai.com/api-keys
3. Thêm vào file `.env`:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
```

**Chi phí ước tính cho OpenAI:**
- GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- GPT-4o: ~$2.50 per 1M input tokens, ~$10.00 per 1M output tokens
- Với chatbot bán hàng, ước tính ~1000 VNĐ/ngày cho 100 conversations

### Option 2: Nâng cấp Google Gemini API

Nâng cấp lên gói trả phí của Google Gemini:

1. Truy cập: https://ai.google.dev/pricing
2. Enable billing cho project
3. Gemini 1.5 Flash có giá rất rẻ: $0.075 per 1M input tokens

**Ưu điểm:**
- Rẻ hơn OpenAI (~50% cheaper)
- Input context window lớn (1M tokens)
- Không cần thay đổi code

### Option 3: Sử dụng cả hai API với fallback

Cấu hình để dùng cả hai API:

```env
# Primary: OpenAI
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini

# Fallback: Gemini
GOOGLE_API_KEY=AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Hệ thống đã được code để ưu tiên Gemini trước, sau đó fallback sang OpenAI. Bạn có thể đổi thứ tự ưu tiên trong `backend/workflow_factory.py`.

### Option 4: Rate limiting và caching

Để giảm số lượng API calls:

1. **Enable caching** - Hệ thống đã có prompt caching built-in
2. **Rate limiting** - Giới hạn số requests từ mỗi user
3. **Smart retry** - Chỉ retry khi thực sự cần thiết

Các tính năng này đã được implement trong:
- `verification/utils/prompt_cache.py`
- `verification/utils/rate_limiter.py`
- `verification/utils/async_retry.py`

## Monitoring API Usage

Để theo dõi API usage:

1. **Google Gemini:**
   - Truy cập: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
   - Xem quota và usage hiện tại

2. **OpenAI:**
   - Truy cập: https://platform.openai.com/usage
   - Xem usage và costs

3. **Application metrics:**
   - Endpoint: `GET /metrics`
   - Xem số requests và active streams

## Testing sau khi fix

Để test xem fix đã hoạt động:

1. **Test với quota exhausted:**
```bash
# Chạy backend
cd backend
python -m uvicorn main:app --reload

# Gửi request từ frontend
# Khi quota hết, user sẽ thấy thông báo thân thiện thay vì escalation không đúng
```

2. **Test với OpenAI API (nếu đã thêm key):**
```bash
# Thêm OPENAI_API_KEY vào .env
# Restart backend
# Test lại - hệ thống sẽ dùng OpenAI thay vì Gemini
```

## Summary

✅ **Đã fix:** Xử lý lỗi API quota đúng cách, không còn escalate nhầm
✅ **Thông báo:** User thấy message thân thiện khi gặp lỗi API
⚠️ **Cần làm:** Thêm OPENAI_API_KEY hoặc upgrade Gemini API để production sử dụng

---

**Liên hệ:** Nếu cần hỗ trợ cấu hình API keys hoặc billing, vui lòng liên hệ team lead.
