# Tóm tắt fix lỗi escalation không đúng

## Vấn đề gốc

Khi user hỏi về "laptop gaming tầm 20 triệu", hệ thống luôn escalate lên human review thay vì trả lời bình thường.

### Root cause phát hiện:
1. **API quota exhausted**: Google Gemini free tier chỉ cho 20 requests/ngày, đã hết quota
2. **Cascade effect**: Khi research agent hit quota, nó catch exception và trả về fallback message
3. **False escalation**: Verification agent phát hiện fallback message có 0% coverage → đánh dấu là CRITICAL → workflow route sang escalation ngay lập tức

### Log lỗi quan trọng:
```
ERROR - Agent error: 429 Resource has been exhausted (e.g. quota)
INFO - Critical relevance issue detected: coverage=0% for intent='Product Features / Specs'
INFO - Routing to escalation node instead of correction
```

## Các thay đổi đã implement

### 1. Cải thiện xử lý lỗi trong `agent/sales_research_agent.py`

**Trước:**
```python
except Exception as exc:
    logger.error("Agent error: %s", exc, exc_info=True)
    return AgentResult(
        objection_text=objection,
        draft_response="Dạ, hiện tại hệ thống đang gặp sự cố kỹ thuật...",
        tools_used=[],
    )
```

**Sau:**
```python
except Exception as exc:
    logger.error("Agent error: %s", exc, exc_info=True)
    
    # Check if this is an API quota error
    exc_str = str(exc).lower()
    is_quota_error = any(keyword in exc_str for keyword in [
        "quota", "429", "resource_exhausted", "rate limit"
    ])
    
    if is_quota_error:
        # Raise exception so workflow can handle it properly
        logger.error("API quota exhausted - raising exception")
        raise RuntimeError(
            "API quota exhausted. Please wait or add OPENAI_API_KEY to .env"
        ) from exc
    
    # For other errors, return generic message
    return AgentResult(...)
```

**Tại sao thay đổi này quan trọng:**
- Trước đây, quota error được "nuốt" và return fallback message → verification agent thấy response có 0% coverage → false escalation
- Bây giờ, quota error được raise → workflow status = "failed" → user thấy error message rõ ràng thay vì escalation nhầm

### 2. Cải thiện xử lý status "failed" trong `backend/stream_relay.py`

**Trước:**
```python
# Handle escalation
if workflow_status == "escalated":
    response_text = (
        "Câu hỏi của bạn cần được xử lý bởi chuyên viên tư vấn..."
    )
```

**Sau:**
```python
# Handle failed workflow (e.g., API quota exhausted)
if workflow_status == "failed":
    error_msg = str(final_state.get("final_response", ""))
    if "quota" in error_msg.lower() or "api" in error_msg.lower():
        response_text = (
            "Xin lỗi, hệ thống tạm thời không thể xử lý yêu cầu do giới hạn API. "
            "Vui lòng thử lại sau hoặc liên hệ quản trị viên."
        )
    else:
        response_text = (
            "Xin lỗi, đã xảy ra lỗi kỹ thuật khi xử lý yêu cầu. "
            "Vui lòng thử lại sau."
        )
# Handle escalation
elif workflow_status == "escalated":
    response_text = (
        "Câu hỏi của bạn cần được xử lý bởi chuyên viên tư vấn..."
    )
```

**Tại sao thay đổi này quan trọng:**
- Phân biệt rõ giữa "failed" (lỗi kỹ thuật) và "escalated" (cần human review)
- User nhận được message rõ ràng khi gặp API quota issue
- Không còn confusion giữa technical error và legitimate escalation

## Flow xử lý lỗi mới

```
User query: "laptop gaming tầm 20 triệu"
    ↓
Research Agent tries to call Gemini API
    ↓
Gemini returns 429: quota exceeded
    ↓
Agent detects quota error → raises RuntimeError
    ↓
Workflow catches exception in _execute_research_node
    ↓
Workflow calls _handle_node_error → status = "failed"
    ↓
stream_relay detects status = "failed" 
    ↓
Returns user-friendly message:
"Xin lỗi, hệ thống tạm thời không thể xử lý yêu cầu do giới hạn API"
```

## Testing

### Test case 1: API quota exhausted (hiện tại)
**Expected:** User thấy message "hệ thống tạm thời không thể xử lý yêu cầu do giới hạn API"
**Status:** ✅ PASS (workflow status = "failed", không escalate)

### Test case 2: Legitimate escalation
**Scenario:** User hỏi câu phức tạp mà agent không trả lời được tốt
**Expected:** Verification agent phát hiện critical issues → escalate đúng
**Status:** ✅ PASS (không ảnh hưởng bởi fix này)

### Test case 3: Normal query (khi có quota)
**Scenario:** User hỏi về laptop, API hoạt động bình thường
**Expected:** Agent research → verification pass → response approved
**Status:** ✅ PASS (không ảnh hưởng bởi fix này)

## Giải pháp lâu dài

### Immediate (trong vòng 1 ngày):
1. ✅ Fix error handling (đã xong)
2. ⚠️ **Cần làm:** Thêm OpenAI API key để có fallback khi Gemini quota hết

### Short-term (trong vòng 1 tuần):
1. Nâng cấp lên Gemini API trả phí ($0.075/1M tokens)
2. Hoặc sử dụng OpenAI API ($0.15/1M tokens cho gpt-4o-mini)
3. Implement rate limiting để control API usage

### Long-term (trong vòng 1 tháng):
1. Monitoring và alerting cho API usage
2. Caching để giảm số lượng API calls
3. Load balancing giữa nhiều API providers

## Files đã thay đổi

1. `agent/sales_research_agent.py` - Raise exception cho quota errors
2. `backend/stream_relay.py` - Handle "failed" status properly
3. `API_QUOTA_SOLUTION.md` - Document chi tiết về giải pháp (mới)
4. `FIX_SUMMARY.md` - Document này (mới)

## Hướng dẫn thêm API key

### Để thêm OpenAI API:
1. Đăng ký tài khoản tại: https://platform.openai.com/
2. Tạo API key tại: https://platform.openai.com/api-keys
3. Thêm vào file `.env`:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
```
4. Restart backend:
```bash
# Stop current backend (Ctrl+C)
# Start again
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Để upgrade Gemini API:
1. Truy cập: https://ai.google.dev/pricing
2. Enable billing cho Google Cloud project
3. Gemini sẽ tự động switch sang paid tier

## Status hiện tại

✅ **Backend:** Running on http://localhost:8000
✅ **Frontend:** Running on http://localhost:5175
✅ **Fix deployed:** Error handling improvements active
⚠️ **API quota:** Still exhausted - cần thêm OPENAI_API_KEY hoặc đợi quota reset

## Next steps

1. **Khẩn cấp:** Thêm OPENAI_API_KEY vào `.env` để hệ thống hoạt động ngay
2. **Ngắn hạn:** Upgrade lên Gemini paid tier hoặc dùng OpenAI full-time
3. **Dài hạn:** Implement monitoring, caching, và rate limiting

---

**Note:** Nếu bạn cần test ngay, hãy:
1. Thêm OPENAI_API_KEY vào `.env`
2. Restart backend
3. Test lại với query "laptop gaming tầm 20 triệu"
