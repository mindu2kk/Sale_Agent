# Hướng Dẫn Học Verification Agent với LangGraph - Giải Thích Chi Tiết Theo Từng File

---

## PHẦN 0 — BỨC TRANH TOÀN CẢNH (Đọc trước khi học bất cứ file nào) ⭐⭐⭐

> Đây là phần QUAN TRỌNG NHẤT. Không có cái nhìn tổng thể này, bạn sẽ đọc code mà không biết mình đang đọc cái gì.

---

### Hệ thống này giải quyết vấn đề gì?

Hãy tưởng tượng bạn là nhân viên bán hàng điện thoại. Khách hỏi: *"iPhone quá đắt, tại sao tôi nên mua?"*

Bạn cần trả lời. Nhưng câu trả lời đó có thể sai theo **3 cách**:

1. **Sai giá** — bạn nói "iPhone 15 giá 35 triệu" nhưng thực tế là 29.9 triệu
2. **Bịa chính sách** — bạn nói "bảo hành 3 năm" nhưng thực tế chỉ 1 năm
3. **Trả lời lạc đề** — khách hỏi về giá nhưng bạn nói về màu sắc

Hệ thống này tự động **phát hiện 3 loại lỗi đó** và **yêu cầu sửa lại** trước khi gửi cho khách.

---

### Hai agent, hai vai trò khác nhau ⭐

```
┌─────────────────────────────────────────────────────────────┐
│                       HỆ THỐNG                              │
│                                                             │
│   [Sales Research Agent]  →→→  [Verification Agent]        │
│        "Viết bản nháp"              "Kiểm tra bản nháp"    │
│                                                             │
│   Nếu sai → Verification Agent tạo feedback cụ thể         │
│   → Sales Agent viết lại → Kiểm tra lại → ...              │
└─────────────────────────────────────────────────────────────┘
```

- **Sales Research Agent** (tuần trước): Nhận câu hỏi của khách, dùng tools tìm thông tin, viết bản nháp trả lời.
- **Verification Agent** (tuần này): Nhận bản nháp đó, kiểm tra 3 tiêu chí, quyết định PASS hay FAIL.

> Điểm mấu chốt: **Verification Agent không viết lại câu trả lời**. Nó chỉ nói "sai chỗ nào, sửa thế nào" rồi Sales Agent tự viết lại.

---

### Tại sao cần LangGraph? ⭐

Tuần trước dùng LangChain — chạy thẳng một lần rồi xong. Tuần này cần **vòng lặp có điều kiện**:

```
Viết → Kiểm tra → Sai → Sửa → Viết lại → Kiểm tra → Đúng → Gửi khách
                    ↑___________________________|
                         (vòng lặp tối đa 3 lần)
```

LangChain không làm được vòng lặp có điều kiện như vậy.
LangGraph làm được vì nó cho phép xây dựng **đồ thị có hướng** (directed graph) với các nhánh điều kiện.

---

### 4 node trong đồ thị ⭐

```
[research] → [verification] → PASS → KẾT THÚC ✅
                    ↓
                   FAIL
                    ↓
             [correction] → [research] (retry)
                    ↓
              QUÁ NHIỀU LỖI / CRITICAL
                    ↓
             [escalation] → KẾT THÚC 🚨 (chuyển người)
```

- **research**: Sales Agent viết bản nháp
- **verification**: Kiểm tra 3 tiêu chí song song
- **correction**: Tạo feedback hướng dẫn sửa
- **escalation**: Khi lỗi quá nghiêm trọng hoặc đã retry quá nhiều lần

---

### Cấu trúc thư mục — mỗi folder làm gì

```
verification/
├── models/      ← Định nghĩa "hình dạng" của dữ liệu (Pydantic)
├── config/      ← Các thông số cấu hình (ngưỡng, timeout, v.v.)
├── agent/       ← Logic kiểm tra thực sự (3 checker + orchestrator)
├── workflow/    ← LangGraph: nodes, routing, correction
└── utils/       ← Công cụ hỗ trợ (cache, logging, v.v.)
```

**Thứ tự phụ thuộc** (cái nào dùng cái nào):

```
config → models → agent/checkers → agent/verification_agent
                                          ↓
                              workflow/correction
                              workflow/routing
                              workflow/workflow  ← entry point
```

> Học theo thứ tự: config → models → agent → workflow.

---

### Tại sao dùng Binary (PASS/FAIL) thay vì điểm số?

Trước đây người ta hay dùng điểm: "câu trả lời này đạt 7/10". Nhưng 7/10 nghĩa là gì? Có gửi cho khách không?

Binary rõ ràng hơn:
- **PASS** → gửi khách ngay
- **FAIL** → sửa lại

Không có vùng xám. Không cần người quyết định "7 điểm có đủ không".

---

## Mục Lục

1. [verification/models/state.py](#1-verificationmodelsstatepy) ⭐⭐⭐
2. [verification/models/verification.py](#2-verificationmodelsverificationpy) ⭐⭐⭐
3. [verification/models/execution.py](#3-verificationmodelsexecutionpy) ⭐⭐
4. [verification/config/config.py](#4-verificationconfigconfigpy) ⭐⭐⭐
5. [verification/config/thresholds_config.py](#5-verificationconfigthresholds_configpy) ⭐⭐
6. [verification/config/config_loader.py](#6-verificationconfigconfig_loaderpy) ⭐
7. [verification/agent/checkers.py](#7-verificationagentcheckerspy) ⭐⭐⭐
8. [verification/agent/verification_agent.py](#8-verificationagentverification_agentpy) ⭐⭐⭐
9. [verification/workflow/routing.py](#9-verificationworkflowroutingpy) ⭐⭐⭐
10. [verification/workflow/correction.py](#10-verificationworkflowcorrectionpy) ⭐⭐⭐
11. [verification/workflow/workflow.py](#11-verificationworkflowworkflowpy) ⭐⭐⭐

---
## 1. `verification/models/state.py` ⭐⭐⭐ QUAN TRỌNG NHẤT

> File này định nghĩa **"tờ giấy"** được truyền qua toàn bộ workflow. Hiểu file này = hiểu dữ liệu chạy trong hệ thống.

---

### Tại sao cần một "tờ giấy" chung?

Hãy nghĩ về một dây chuyền sản xuất. Mỗi trạm làm một việc, nhưng tất cả đều làm việc trên **cùng một sản phẩm**. Sản phẩm đó được truyền từ trạm này sang trạm khác, mỗi trạm thêm thông tin vào.

Trong LangGraph, "sản phẩm" đó là `WorkflowState` — một dictionary chứa tất cả thông tin của workflow tại một thời điểm.

```
research_node đọc state → thêm draft_response → trả về state mới
verification_node đọc state → thêm verification_result → trả về state mới
correction_node đọc state → thêm correction_feedback → trả về state mới
```

Không node nào "nhớ" thông tin riêng. Tất cả đều nằm trong state.

---

### `WorkflowState` (TypedDict) ⭐⭐⭐

**Đây là class quan trọng nhất trong toàn bộ hệ thống.**

```python
class WorkflowState(TypedDict):
    # ── NHÓM 1: INPUT ──────────────────────────────────────────
    objection_text: str          # "iPhone quá đắt" ← đầu vào từ khách
    customer_context: Optional[Dict]  # Thông tin thêm về khách (không bắt buộc)

    # ── NHÓM 2: SAU KHI RESEARCH NODE CHẠY ────────────────────
    draft_response: str          # "iPhone 15 giá 35M VND..." ← Sales Agent viết
    tools_used: List[str]        # ["product_search", "price_lookup"] ← tools đã dùng
    research_reasoning: str      # Lý do agent chọn thông tin này
    research_sources: List[str]  # Nguồn dữ liệu đã dùng

    # ── NHÓM 3: SAU KHI VERIFICATION NODE CHẠY ────────────────
    verification_result: Optional[VerificationResult]  # Kết quả kiểm tra

    # ── NHÓM 4: SAU KHI CORRECTION NODE CHẠY ──────────────────
    correction_feedback: Optional[str]  # Hướng dẫn sửa lỗi chi tiết
    retry_count: int             # Đã thử mấy lần (bắt đầu từ 0)
    max_retries: int             # Tối đa bao nhiêu lần (thường là 3)

    # ── NHÓM 5: KẾT QUẢ CUỐI ──────────────────────────────────
    final_response: str          # Câu trả lời đã được duyệt (chỉ có khi PASS)
    workflow_status: Literal[...]  # Đang ở bước nào (xem bên dưới)

    # ── NHÓM 6: TRACKING & OBSERVABILITY ──────────────────────
    execution_log: List[ExecutionStep]  # Lịch sử từng bước đã chạy
    start_time: str              # Bắt đầu lúc nào (ISO format)
    end_time: Optional[str]      # Kết thúc lúc nào
    resource_usage: Dict         # Tốn bao nhiêu CPU/memory/token/tiền
    error_log: List[Dict]        # Lỗi nào đã xảy ra
    workflow_id: str             # ID duy nhất: "wf_20240115_103000_abc12345"
    correlation_id: str          # ID cho distributed tracing
    config: Dict[str, Any]       # Snapshot config tại thời điểm chạy
```

**Tại sao dùng TypedDict chứ không phải Pydantic?**

LangGraph yêu cầu state phải là dict-like (có thể truy cập bằng `state["key"]`).
- `TypedDict` = dict nhưng có type hints → Python biết kiểu dữ liệu nhưng không tự validate
- `Pydantic BaseModel` = validate tự động nhưng không phải dict thuần

Giải pháp của hệ thống: dùng `TypedDict` cho LangGraph, tạo thêm `WorkflowStateValidator` (Pydantic) để validate khi cần.

---

### `workflow_status` — Vòng đời của workflow ⭐⭐⭐

Đây là trường quan trọng nhất để biết workflow đang ở đâu:

```
"initialized"   → Vừa tạo xong, chưa làm gì
      ↓
"researching"   → Research node đang chạy (Sales Agent đang viết)
      ↓
"verifying"     → Verification node đang chạy (đang kiểm tra)
      ↓
    ┌─────────────────────────────────────────┐
    │                                         │
"approved"      "correcting"           "escalated"
(terminal ✅)        ↓                  (terminal ��)
                "researching"
                (retry loop)
                     ...
                "approved" hoặc "escalated" hoặc "failed"
```

**3 trạng thái terminal** (điểm kết thúc, không thể tiếp tục):
- `"approved"` → Câu trả lời đã được duyệt, gửi cho khách
- `"escalated"` → Lỗi quá nghiêm trọng, chuyển cho người xử lý
- `"failed"` → Workflow bị lỗi kỹ thuật

---

### `resource_usage` — Tại sao cần track tài nguyên? ⭐

```python
resource_usage: Dict[str, Any] = {
    "cpu_time_seconds": 0.0,    # Tốn bao nhiêu CPU
    "memory_peak_mb": 0.0,      # Tốn bao nhiêu RAM
    "llm_tokens_total": 0,      # ← QUAN TRỌNG: token = tiền thật
    "llm_cost_usd": 0.0,        # ← Chi phí thực tế bằng USD
    "db_queries_count": 0,      # Số lần query database
    "cache_hits": 0,            # Bao nhiêu lần lấy từ cache (nhanh, miễn phí)
    "cache_misses": 0           # Bao nhiêu lần phải query thật (chậm, tốn tiền)
}
```

`llm_tokens_total` và `llm_cost_usd` đặc biệt quan trọng vì mỗi lần gọi LLM tốn tiền thật.

`cache_hits` vs `cache_misses`: nếu cache hit rate thấp (nhiều miss), nghĩa là cache không hiệu quả.

Ví dụ: nếu 1 workflow tốn $0.05 và hệ thống xử lý 1000 câu hỏi/ngày → $50/ngày. Tối ưu cache có thể giảm xuống $20/ngày.

---

### `WorkflowStateValidator` — Pydantic để validate ⭐⭐

Vì TypedDict không tự validate, file tạo thêm class Pydantic:

```python
class WorkflowStateValidator(BaseModel):
    objection_text: str = Field(min_length=10, max_length=5000)
    retry_count: int = Field(ge=0, le=10, default=0)
    # ... tất cả các field khác với ràng buộc
```

**4 validator quan trọng:**

**1. `validate_start_time`** — đảm bảo timestamp đúng format:
```python
@validator('start_time', pre=True, always=True)
def validate_start_time(cls, v):
    if v is None:
        return datetime.now().isoformat()  # Tự tạo nếu không có
    try:
        datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v
    except ValueError:
        raise ValueError("start_time must be valid ISO format timestamp")
```
→ Nếu không truyền `start_time`, tự động lấy thời gian hiện tại.

**2. `validate_end_time`** — đảm bảo logic thời gian:
```python
@validator('end_time')
def validate_end_time(cls, v, values):
    if end_dt <= start_dt:
        raise ValueError("end_time must be after start_time")
```
→ Không thể kết thúc trước khi bắt đầu.

**3. `validate_retry_count`** — đảm bảo không vượt giới hạn:
```python
@validator('retry_count')
def validate_retry_count(cls, v, values):
    max_retries = values.get('max_retries', 3)
    if v > max_retries:
        raise ValueError(f"retry_count ({v}) cannot exceed max_retries ({max_retries})")
```
→ Nếu `max_retries=3`, không thể có `retry_count=5`.

**4. `validate_resource_usage_structure`** — đảm bảo đủ các field:
```python
required_fields = [
    "cpu_time_seconds", "memory_peak_mb",
    "llm_tokens_total", "llm_cost_usd",
    "db_queries_count", "cache_hits", "cache_misses"
]
# Nếu thiếu field nào → tự thêm với giá trị 0
```
→ Đảm bảo `resource_usage` luôn có đủ cấu trúc để tính toán.

**Computed properties hữu ích:**

```python
@property
def is_terminal_state(self) -> bool:
    # Kiểm tra workflow đã kết thúc chưa
    return self.workflow_status in ["approved", "escalated", "failed"]

@property
def cache_hit_rate(self) -> float:
    # Tính tỷ lệ cache hit
    hits = self.resource_usage.get("cache_hits", 0)
    misses = self.resource_usage.get("cache_misses", 0)
    total = hits + misses
    return hits / total if total > 0 else 0.0

@property
def has_critical_issues(self) -> bool:
    # Kiểm tra có lỗi nghiêm trọng không
    if not self.verification_result:
        return False
    return self.verification_result.criteria.critical_issues_count > 0
```

---

### `WorkflowConfig` — Config nằm trong file state ⭐⭐

File này cũng chứa `WorkflowConfig` — class config cho toàn bộ workflow:

```python
class WorkflowConfig(BaseModel):
    # Ngưỡng kiểm tra
    price_tolerance_percent: float = 1.0   # Sai <=1% = OK (vì làm tròn số)
    relevance_min_coverage: float = 0.7    # Cover >=70% câu hỏi = OK
    max_retries: int = 3                   # Tối đa 3 lần sửa

    # Hiệu năng
    parallel_verification: bool = True     # Chạy 3 checker cùng lúc (nhanh hơn 3x)
    early_termination: bool = True         # Dừng ngay khi gặp CRITICAL (tiết kiệm API)
    async_timeout_seconds: int = 30        # Timeout toàn bộ verification

    # Cache
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600          # Cache sống 1 giờ
    cache_max_size: int = 1000             # Tối đa 1000 entries

    # LLM
    llm_temperature: float = 0.1           # Thấp = kết quả nhất quán, ít ngẫu nhiên
```

**Tại sao `price_tolerance_percent = 1.0`?**

Giá trong database có thể có sai số nhỏ do làm tròn. Nếu đặt tolerance = 0%, thì "29,990,000" và "29,990,001" sẽ bị coi là sai. Tolerance 1% cho phép sai số nhỏ mà không báo lỗi.

**Tại sao `llm_temperature = 0.1`?**

Temperature cao (gần 1.0) → LLM sáng tạo, ngẫu nhiên → cùng input có thể cho output khác nhau.
Temperature thấp (gần 0.0) → LLM nhất quán → cùng input luôn cho output giống nhau.

Với verification, chúng ta cần **nhất quán**: cùng một draft phải luôn cho cùng kết quả PASS/FAIL.

---

### `create_initial_workflow_state()` — Hàm khởi tạo ⭐⭐

```python
def create_initial_workflow_state(
    objection_text: str,
    config: Optional[WorkflowConfig] = None,
    customer_context: Optional[Dict] = None
) -> Dict[str, Any]:
```

Đây là **điểm bắt đầu của mọi workflow**. Hàm này:
1. Tạo `workflow_id` unique (dạng `wf_20240115_103000_abc12345`)
2. Tạo `correlation_id` cho distributed tracing
3. Set tất cả field về giá trị mặc định (draft_response = "", retry_count = 0, v.v.)
4. Validate bằng `WorkflowStateValidator` → báo lỗi ngay nếu input không hợp lệ
5. Trả về dict đã validate

**Tại sao validate ngay từ đầu?**

Nếu `objection_text` quá ngắn (< 10 ký tự) hoặc quá dài (> 5000 ký tự), tốt hơn là báo lỗi ngay lúc khởi tạo thay vì để lỗi xảy ra ở giữa workflow (khó debug hơn nhiều).

---

### Tóm tắt file state.py

`state.py` định nghĩa **bộ nhớ chung** của workflow. Mỗi node đọc state, làm việc, cập nhật state, rồi truyền cho node tiếp theo.

Điều này quan trọng vì:
- **Dễ debug**: nhìn vào state là biết workflow đang ở đâu, đã làm gì
- **Dễ resume**: nếu crash, load state từ checkpoint là chạy tiếp được
- **Dễ test**: tạo state giả, chạy một node, kiểm tra state output

---
## 2. `verification/models/verification.py` ⭐⭐⭐ QUAN TRỌNG

> File này định nghĩa **kết quả của quá trình kiểm tra** — từ từng lỗi nhỏ đến kết quả tổng thể. Đây là "ngôn ngữ" mà Verification Agent dùng để nói chuyện với phần còn lại của hệ thống.

---

### Tổng quan: Các class trong file này liên hệ với nhau như thế nào?

```
VerificationResult          ← Kết quả tổng thể (1 lần verification)
    └── criteria: RubricCriteria   ← Kết quả 3 tiêu chí
            ├── price_issues: List[PriceIssue]      ← Danh sách lỗi giá
            ├── policy_issues: List[PolicyIssue]    ← Danh sách lỗi chính sách
            └── relevance_issues: List[RelevanceIssue] ← Danh sách lỗi chủ đề

IssueSeverity (Enum)        ← CRITICAL / MAJOR / MINOR (dùng trong tất cả issue)
FeedbackReport              ← Đóng gói feedback để inject vào Research Agent
FailedCriterion             ← Mô tả một tiêu chí bị fail
```

---

### `IssueSeverity` — Phân loại mức độ nghiêm trọng ⭐⭐⭐

```python
class IssueSeverity(str, Enum):
    CRITICAL = "critical"  # Escalate ngay, KHÔNG retry
    MAJOR = "major"        # Cần sửa, CÓ THỂ retry
    MINOR = "minor"        # Nên sửa, ít ảnh hưởng
```

**Tại sao cần 3 mức?**

Không phải lỗi nào cũng như nhau:
- Giá sai 0.5% → MINOR (làm tròn số, chấp nhận được)
- Giá sai 20% → MAJOR (sai đáng kể, cần sửa)
- Giá sai 50% → CRITICAL (sai nghiêm trọng, không thể gửi khách)
- Bịa chính sách bảo hành → CRITICAL (rủi ro pháp lý, escalate ngay)

**Mức độ ảnh hưởng đến workflow:**
- `CRITICAL` → Routing sẽ escalate ngay (bỏ qua retry)
- `MAJOR` → Routing sẽ correction → retry
- `MINOR` → Routing sẽ correction → retry (nhưng ít urgent hơn)

---

### `PriceIssue` — Model cho một lỗi giá ⭐⭐⭐

```python
class PriceIssue(BaseModel):
    product_name: str           # "iPhone 15 Pro Max"
    product_sku: Optional[str]  # "IP15PM-256" (mã sản phẩm để tra cứu)
    mentioned_price: str        # "35,000,000 VND" (giá trong draft — SAI)
    actual_price: str           # "29,990,000 VND" (giá thực từ DB — ĐÚNG)
    deviation_percent: float    # 16.7 (% sai lệch)
    currency: str               # "VND"
    severity: IssueSeverity     # MINOR/MAJOR/CRITICAL
    explanation: str            # "Giá sai lệch 16.7%, vượt ngưỡng 1%"
    correction_suggestion: str  # "Cập nhật giá thành 29,990,000 VND (SKU: IP15PM-256)"
```

**Tại sao cần `correction_suggestion`?**

Không chỉ nói "sai" mà còn nói "sửa thế nào". Điều này giúp Research Agent biết chính xác cần làm gì khi retry, thay vì phải đoán.

**Property `is_critical_deviation`:**
```python
@property
def is_critical_deviation(self) -> bool:
    return self.deviation_percent is not None and self.deviation_percent > 30.0
```
→ Nếu giá sai hơn 30%, đây là lỗi nghiêm trọng cần escalate ngay.

---

### `PolicyIssue` — Model cho một lỗi chính sách ⭐⭐⭐

```python
class PolicyIssue(BaseModel):
    mentioned_policy: str       # "Bảo hành 3 năm cho tất cả sản phẩm"
    policy_type: Literal[       # Loại chính sách
        "warranty",             # Bảo hành
        "return",               # Đổi trả
        "exchange",             # Đổi máy
        "service",              # Sửa chữa
        "support"               # Hỗ trợ
    ]
    is_fabricated: bool         # True = bịa đặt hoàn toàn (không có trong DB)
    is_inaccurate: bool         # True = có trong DB nhưng sai nội dung
    is_incomplete: bool         # True = thiếu thông tin quan trọng
    correct_policy: Optional[str]  # Chính sách đúng từ tài liệu chính thức
    severity: IssueSeverity
    explanation: str
    source_document: Optional[str]  # "warranty_policy_2024.pdf"
```

**Phân biệt 3 loại lỗi chính sách:**

| Loại | Ý nghĩa | Severity |
|------|---------|----------|
| `is_fabricated=True` | Chính sách không tồn tại trong DB | CRITICAL |
| `is_inaccurate=True` | Chính sách có trong DB nhưng nội dung sai | MAJOR |
| `is_incomplete=True` | Chính sách đúng nhưng thiếu chi tiết | MINOR |

**Tại sao `is_fabricated` là CRITICAL?**

Nếu Sales Agent nói "bảo hành 3 năm" nhưng thực tế chỉ 1 năm, khách hàng sẽ khiếu nại. Đây là rủi ro pháp lý và uy tín nghiêm trọng → phải escalate cho người xử lý ngay.

**Property `requires_immediate_escalation`:**
```python
@property
def requires_immediate_escalation(self) -> bool:
    return self.is_fabricated and self.severity == IssueSeverity.CRITICAL
```

---

### `RelevanceIssue` — Model cho lỗi không đúng chủ đề ⭐⭐

```python
class RelevanceIssue(BaseModel):
    objection_intent: str       # "So sánh giá iPhone vs Samsung"
    detected_intents: List[str] # ["price_comparison", "value_justification"]
    response_coverage: float    # 0.6 = chỉ trả lời được 60% câu hỏi
    missing_aspects: List[str]  # ["camera comparison", "gaming performance"]
    off_topic_content: List[str] # ["Apple history", "irrelevant info"]
    empathy_score: float        # 0.3 = thiếu cảm thông với khách
    severity: IssueSeverity
    explanation: str
```

**`response_coverage` là gì?**

Nếu khách hỏi 5 điều (giá, camera, pin, bảo hành, so sánh Samsung) mà response chỉ trả lời 3 điều → coverage = 3/5 = 0.6 (60%).

**`empathy_score` là gì?**

Đo mức độ "cảm thông" trong câu trả lời. Khách phàn nàn "iPhone quá đắt" — response tốt nên thừa nhận concern của khách trước khi giải thích, thay vì chỉ liệt kê tính năng lạnh lùng.

---

### `RubricCriteria` — Tổng hợp kết quả 3 tiêu chí ⭐⭐⭐

```python
class RubricCriteria(BaseModel):
    # Kết quả binary của 3 tiêu chí
    price_accuracy_pass: bool       # Giá có đúng không?
    policy_authenticity_pass: bool  # Chính sách có xác thực không?
    topic_relevance_pass: bool      # Có đúng chủ đề không?

    # Danh sách issues chi tiết
    price_issues: List[PriceIssue]
    policy_issues: List[PolicyIssue]
    relevance_issues: List[RelevanceIssue]

    # Computed trong __init__ (tự tính, không cần truyền vào)
    overall_pass: Optional[bool]        # = price AND policy AND relevance
    critical_issues_count: Optional[int] # Đếm tổng CRITICAL issues
```

**Logic tính `overall_pass` — QUAN TRỌNG ⭐⭐⭐**

```python
def __init__(self, **data):
    super().__init__(**data)
    # Tự tính overall_pass sau khi khởi tạo
    if self.overall_pass is None:
        self.overall_pass = (
            self.price_accuracy_pass and
            self.policy_authenticity_pass and
            self.topic_relevance_pass
        )
```

Đây là **AND logic**: tất cả 3 tiêu chí phải PASS thì mới overall PASS.

Ví dụ:
- price=PASS, policy=PASS, relevance=FAIL → overall=FAIL
- price=PASS, policy=FAIL, relevance=PASS → overall=FAIL
- price=PASS, policy=PASS, relevance=PASS → overall=PASS

**Tại sao dùng Python thuần thay vì AI để tính `overall_pass`?**

Vì đây là logic đơn giản, deterministic. Dùng AI để tính `True AND True AND False` là lãng phí token và có thể cho kết quả không nhất quán.

**`get_escalation_priority()` — Quyết định mức độ ưu tiên escalate:**

```python
def get_escalation_priority(self) -> Literal["immediate", "high", "medium", "low"]:
    if self.critical_issues_count >= 3:
        return "immediate"
    elif self.critical_issues_count >= 1:
        # Kiểm tra thêm: có bịa chính sách không?
        fabricated_policies = any(
            issue.is_fabricated for issue in self.policy_issues
            if issue.severity == IssueSeverity.CRITICAL
        )
        if fabricated_policies:
            return "immediate"  # Bịa chính sách = escalate ngay
        return "high"
    elif self.get_major_issues_count() >= 2:
        return "medium"
    else:
        return "low"
```

---

### `VerificationResult` — Kết quả cuối cùng của một lần verification ⭐⭐⭐

```python
class VerificationResult(BaseModel):
    criteria: RubricCriteria        # Kết quả 3 tiêu chí (xem trên)
    timestamp: datetime             # Thời điểm verification
    verification_reasoning: str     # Lý do quyết định (min 10 ký tự)
    execution_time_seconds: float   # Mất bao lâu (giây)
    llm_tokens_used: int            # Tốn bao nhiêu token
    step_latencies: Optional[Dict]  # Latency từng checker riêng lẻ
    has_critical_issues: bool       # Có CRITICAL issue không
    immediate_termination: bool     # Có cần dừng workflow ngay không
```

**Properties quan trọng:**

```python
@property
def is_approved(self) -> bool:
    # Đây là câu hỏi cuối cùng: "Có gửi cho khách không?"
    return self.criteria.overall_pass  # True hoặc False, không có vùng xám

@property
def requires_correction(self) -> bool:
    return not self.criteria.overall_pass  # Ngược lại với is_approved

@property
def requires_escalation(self) -> bool:
    # Cần chuyển cho người xử lý không?
    return (
        self.criteria.critical_issues_count > 0 or
        self.criteria.get_escalation_priority() in ["immediate", "high"]
    )
```

**`get_correction_feedback()` — Method quan trọng nhất ⭐⭐⭐**

Method này sinh ra đoạn text hướng dẫn sửa lỗi, được inject vào prompt của Research Agent khi retry:

```python
def get_correction_feedback(self) -> str:
    if self.is_approved:
        return "✅ No corrections needed - verification passed"

    feedback_parts = [
        "🔄 VERIFICATION FAILED - Corrections needed:",
        self.criteria.get_failure_summary(),
        f"📊 Issue Summary: {critical} critical, {major} major, {minor} minor",
        f"⚠️ Escalation Priority: {self.escalation_priority.upper()}",
    ]

    # Thêm chi tiết từng loại lỗi
    if self.criteria.price_issues:
        feedback_parts.append("💰 PRICE ACCURACY ISSUES:")
        for issue in self.criteria.price_issues:
            feedback_parts.append(f"  - {issue.explanation}")
            feedback_parts.append(f"    💡 Suggestion: {issue.correction_suggestion}")

    # ... tương tự cho policy và relevance issues

    return "\n".join(feedback_parts)
```

Output của method này trông như thế này:
```
🔄 VERIFICATION FAILED - Corrections needed:
🔄 Verification FAILED: ❌ Price Accuracy
📊 Issue Summary: 0 critical, 1 major, 0 minor
⚠️ Escalation Priority: MEDIUM

💰 PRICE ACCURACY ISSUES:
  - Price deviation 16.7% exceeds tolerance 1% for iPhone 15 Pro Max
    💡 Suggestion: Update price to 29,990,000 VND (SKU: IP15PM-256)
    🔍 Verify SKU: IP15PM-256
```

Đây chính là thứ được inject vào prompt của Sales Agent khi retry.

---

### `FeedbackReport` và `FailedCriterion` — Hai model phụ

Hai model này ít quan trọng hơn, chỉ dùng để đóng gói feedback có cấu trúc hơn:

- `FailedCriterion`: Mô tả một tiêu chí bị fail với `criterion_name`, `explanation`, `correction_suggestions`
- `FeedbackReport`: Gộp tất cả `FailedCriterion` lại, có sẵn `correction_prompt` để inject vào Research Agent

Điểm đặc biệt: `FeedbackReport.correction_prompt` được tạo bằng template, **không cần gọi LLM** → nhanh và rẻ hơn.

---

### Tóm tắt file verification.py

File này định nghĩa "ngôn ngữ" của verification:
- `IssueSeverity`: Mức độ nghiêm trọng (CRITICAL/MAJOR/MINOR)
- `PriceIssue`, `PolicyIssue`, `RelevanceIssue`: Mô tả từng lỗi cụ thể
- `RubricCriteria`: Tổng hợp kết quả 3 tiêu chí
- `VerificationResult`: Kết quả cuối cùng với `is_approved` và `get_correction_feedback()`

---
## 3. `verification/models/execution.py` ⭐⭐

> File này định nghĩa models để **theo dõi hiệu năng và lịch sử thực thi**. Ít quan trọng hơn state.py và verification.py, nhưng cần hiểu để debug và tối ưu hệ thống.

---

### Tại sao cần track execution?

Khi hệ thống chạy trong production, bạn cần biết:
- Workflow nào đang chạy? Đang ở bước nào?
- Bước nào chậm nhất? Tốn bao nhiêu tiền?
- Có bao nhiêu lỗi? Lỗi ở đâu?

Không có execution tracking, bạn sẽ "mù" khi hệ thống có vấn đề.

---

### `ExecutionStep` — Ghi lại một bước thực thi ⭐⭐

Mỗi khi một node trong workflow hoàn thành, nó tạo một `ExecutionStep` và append vào `state["execution_log"]`.

```python
class ExecutionStep(BaseModel):
    # Thông tin cơ bản
    timestamp: str              # "2024-01-15T10:30:15.123Z"
    node_name: str              # "research", "verification", "correction"
    execution_time: float       # 2.5 (giây)
    status: ExecutionStatus     # SUCCESS / FAILED / TIMEOUT / RETRY / SKIPPED

    # Tóm tắt input/output (max 200 ký tự để tiết kiệm bộ nhớ)
    input_summary: str          # "draft: 'iPhone 15 Pro Max có giá...'"
    output_summary: str         # "verification_result: PASS (no issues)"

    # Lỗi (nếu có)
    error_details: Optional[str]  # Chi tiết lỗi
    error_type: Optional[str]     # "timeout", "api_error", "validation_error"

    # Metrics hiệu năng
    metrics: Dict[str, Any]     # Custom metrics (db_queries, cache_hits, v.v.)
    memory_usage_mb: float      # RAM đã dùng
    cpu_usage_percent: float    # CPU đã dùng

    # Tracing
    correlation_id: str         # ID để trace qua nhiều services
    parent_correlation_id: str  # ID của workflow cha
    workflow_id: str            # ID của workflow này

    # LLM metrics
    llm_tokens_input: int       # Token gửi đi
    llm_tokens_output: int      # Token nhận về
    llm_cost_usd: float         # Chi phí ước tính
```

**Ví dụ thực tế:**

Sau khi verification node chạy xong, nó tạo:
```python
ExecutionStep(
    node_name="verification",
    execution_time=2.5,
    status=ExecutionStatus.SUCCESS,
    input_summary="draft: 'iPhone 15 Pro Max có giá 35M...'",
    output_summary="result: FAILED (critical=0, price=FAIL, policy=PASS, relevance=PASS)",
    metrics={
        "overall_pass": False,
        "critical_issues": 0,
        "price_accuracy_pass": False,
        "tokens_used": 1250,
        "early_termination_triggered": False,
    },
    llm_tokens_input=850,
    llm_tokens_output=400,
    llm_cost_usd=0.0125
)
```

---

### `WorkflowMetrics` — Tổng hợp metrics toàn workflow ⭐

```python
class WorkflowMetrics(BaseModel):
    # Thời gian
    total_execution_time: float     # Tổng thời gian
    average_step_time: float        # Trung bình mỗi bước (tự tính)
    min_step_time: float            # Bước nhanh nhất
    max_step_time: float            # Bước chậm nhất

    # Retry & Success
    total_retries: int              # Tổng số lần retry
    total_steps: int                # Tổng số bước đã chạy
    successful_steps: int           # Số bước thành công
    failed_steps: int               # Số bước thất bại
    success_rate: float             # Tự tính: successful/total

    # Issues
    critical_issues_found: int
    major_issues_found: int
    minor_issues_found: int

    # Chi phí
    llm_tokens_used: int
    cost_estimate: float            # USD

    # Cache
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float           # Tự tính: hits/(hits+misses)

    # Chất lượng
    verification_pass_rate: float   # Tỷ lệ verification pass
    escalation_rate: float          # Tỷ lệ phải escalate
```

**Property `performance_grade`:**
```python
@property
def performance_grade(self) -> str:
    if self.success_rate >= 0.9 and self.total_retries < 2:
        return "A"  # Xuất sắc
    elif self.success_rate >= 0.8 and self.total_retries < 3:
        return "B"  # Tốt
    elif self.success_rate >= 0.7 and self.total_retries < 4:
        return "C"  # Trung bình
    # ...
```

**`get_optimization_recommendations()`:**
```python
def get_optimization_recommendations(self) -> List[str]:
    recommendations = []
    if self.cache_hit_rate < 0.7:
        recommendations.append("🎯 Improve caching - hit rate below 70%")
    if self.average_step_time > 5.0:
        recommendations.append("⚡ Optimize step execution - average time > 5s")
    if self.cost_per_success > 0.05:
        recommendations.append("💰 Optimize LLM usage - cost per success > $0.05")
    return recommendations
```

---

### `WorkflowExecutionLog` và `WorkflowTracker` — Ít quan trọng hơn

- **`WorkflowExecutionLog`**: Log đầy đủ của một workflow execution (steps + errors + warnings + node history). Dùng để debug sau khi có vấn đề.

- **`WorkflowTracker`**: Singleton theo dõi tất cả active workflows đang chạy. Phát hiện slow workflows và kiểm tra system overload. Dùng cho monitoring real-time.

---

## 4. `verification/config/config.py` ⭐⭐⭐ QUAN TRỌNG

> File này định nghĩa **tất cả thông số cấu hình** của Verification Agent. Đây là nơi bạn điều chỉnh hành vi của hệ thống mà không cần sửa code.

---

### Tại sao cần config riêng?

Không nên hardcode các con số như `price_tolerance = 1.0` trực tiếp trong code vì:
- Muốn thay đổi phải sửa code → deploy lại
- Môi trường khác nhau cần giá trị khác nhau (dev vs production)
- Không thể thay đổi khi hệ thống đang chạy

Config file giải quyết tất cả vấn đề này.

---

### `VerificationConfig` — Class config trung tâm ⭐⭐⭐

Class này được **inject vào hầu hết các component** trong hệ thống. Hiểu class này = hiểu cách hệ thống hoạt động.

**Nhóm 1: Verification Thresholds (Ngưỡng kiểm tra)**

```python
# Giá
price_tolerance_percent: float = 1.0    # Sai <=1% = PASS (vì làm tròn số)
price_critical_threshold: float = 30.0  # Sai >30% = CRITICAL (escalate ngay)

# Chính sách
policy_citation_required: bool = True   # Phải có trích dẫn nguồn
policy_forbidden_phrases: List[str] = [
    "tự bịa", "không có trong hệ thống",
    "fabricated", "made up", "personal opinion"
]  # Nếu draft chứa những từ này → CRITICAL

# Chủ đề
relevance_min_coverage: float = 0.7     # Phải cover >=70% câu hỏi
relevance_empathy_bonus: bool = True    # Có empathy → bonus điểm
```

**Nhóm 2: Workflow Settings (Cài đặt workflow)**

```python
max_retries: int = 3                    # Tối đa 3 lần sửa trước khi escalate
critical_issue_escalation: bool = True  # Gặp CRITICAL → escalate ngay (bỏ qua retry)
retry_backoff_seconds: float = 1.0      # Chờ 1 giây giữa các lần retry
```

**Nhóm 3: Performance (Hiệu năng)**

```python
async_timeout_seconds: int = 30         # Timeout toàn bộ verification
parallel_verification: bool = True      # Chạy 3 checker cùng lúc (nhanh 3x)
early_termination: bool = True          # Dừng ngay khi gặp CRITICAL (tiết kiệm API)
enable_caching: bool = True             # Bật cache
cache_ttl_seconds: int = 3600           # Cache sống 1 giờ
cache_max_size: int = 1000              # Tối đa 1000 entries trong cache
```

**Nhóm 4: LLM Settings**

```python
llm_model_name: str = "gpt-4"          # Model LLM dùng
llm_temperature: float = 0.1           # Thấp = nhất quán, ít ngẫu nhiên
llm_max_tokens: int = 2000             # Tối đa 2000 token mỗi lần gọi
max_cost_per_verification: float = 0.05 # Tối đa $0.05 mỗi lần verify
```

**Nhóm 5: Observability (Quan sát)**

```python
log_level: LogLevel = LogLevel.INFO     # Mức độ log
detailed_logging: bool = True           # Log chi tiết
performance_tracking: bool = True       # Track hiệu năng
enable_metrics_export: bool = True      # Export metrics ra ngoài
```

**Nhóm 6: Security (Bảo mật)**

```python
max_objection_length: int = 5000        # Giới hạn độ dài input
max_draft_length: int = 10000           # Giới hạn độ dài draft
rate_limit_per_minute: int = 60         # Tối đa 60 verification/phút
```

---

### Validator quan trọng ⭐

```python
@validator('price_critical_threshold')
def validate_price_thresholds(cls, v, values):
    tolerance = values.get('price_tolerance_percent', 1.0)
    if v <= tolerance:
        raise ValueError("Critical threshold must be greater than tolerance")
    return v
```

Validator này đảm bảo: `critical_threshold > tolerance`.

Tại sao? Nếu `tolerance = 5%` và `critical_threshold = 3%`, thì một lỗi 4% sẽ vừa PASS (vì < tolerance) vừa CRITICAL (vì > critical_threshold) — mâu thuẫn logic.

---

### Các hàm load config ⭐⭐

**`load_config_from_yaml()` — Hàm quan trọng nhất:**

```python
def load_config_from_yaml(config_dir=None, environment=None) -> VerificationConfig:
    # Thứ tự ưu tiên (sau ghi đè trước):
    # 1. Default Pydantic values (thấp nhất)
    # 2. verification_config.yaml
    # 3. environments/{environment}.yaml
    # 4. Environment variables VERIFICATION_* (cao nhất)
```

Ví dụ thực tế:
- `verification_config.yaml` có `price_tolerance_percent: 1.0`
- `environments/production.yaml` có `price_tolerance_percent: 0.5`
- Env var `VERIFICATION_PRICE_TOLERANCE_PERCENT=0.3`

Kết quả cuối: `price_tolerance_percent = 0.3` (env var thắng)

**`get_config()` — Singleton pattern:**

```python
_global_config: Optional[VerificationConfig] = None

def get_config() -> VerificationConfig:
    global _global_config
    if _global_config is None:
        _global_config = load_config_from_yaml()  # Load lần đầu
    return _global_config  # Các lần sau trả về cached
```

Singleton đảm bảo toàn bộ hệ thống dùng cùng một config instance.

**`reload(environment)` — Reload không cần restart:**

```python
def reload(environment=None) -> VerificationConfig:
    global _global_config
    _global_config = load_config_from_yaml(environment=environment)
    return _global_config
```

Dùng khi cần thay đổi config trong production mà không muốn restart server.

---
## 5. `verification/config/thresholds_config.py` ⭐⭐

> File này định nghĩa **ngưỡng chi tiết hơn** cho từng loại kiểm tra. Tách biệt khỏi `VerificationConfig` để dễ tune từng phần mà không ảnh hưởng phần khác.

---

### Tại sao tách thành file riêng?

`VerificationConfig` chứa tất cả config của hệ thống. Nhưng các ngưỡng kiểm tra (thresholds) rất hay thay đổi khi tune hệ thống. Tách ra file riêng giúp:
- Dễ tìm và sửa
- Có thể load riêng mà không cần load toàn bộ config
- Có validators riêng đảm bảo logic nhất quán

---

### `PriceAccuracyThresholds` ⭐⭐

```python
class PriceAccuracyThresholds(BaseModel):
    pass_tolerance_percent: float = 1.0    # <=1% → PASS (không báo lỗi)
    minor_threshold_percent: float = 5.0   # 1-5% → MINOR issue
    major_threshold_percent: float = 15.0  # 5-15% → MAJOR issue
    critical_threshold_percent: float = 30.0  # >30% → CRITICAL issue
    missing_price_severity: IssueSeverity = MAJOR  # Không có giá → MAJOR
```

**Cách phân loại deviation:**

```
deviation = 0.5%  → PASS (trong tolerance)
deviation = 3%    → MINOR (1-5%)
deviation = 10%   → MAJOR (5-15%)
deviation = 40%   → CRITICAL (>30%)
```

**Method `classify_price_deviation()`:**

```python
def classify_price_deviation(self, deviation_percent: float) -> IssueSeverity:
    if deviation_percent >= self.critical_threshold_percent:  # >= 30%
        return IssueSeverity.CRITICAL
    if deviation_percent >= self.minor_threshold_percent:     # >= 5%
        return IssueSeverity.MAJOR
    return IssueSeverity.MINOR                                # < 5%
```

**Validators đảm bảo thứ tự logic:**

```python
@validator("major_threshold_percent")
def major_must_exceed_minor(cls, v, values):
    minor = values.get("minor_threshold_percent", 0)
    if v <= minor:
        raise ValueError(f"major ({v}) must be > minor ({minor})")
    return v
```

Không thể có `major_threshold < minor_threshold` — sẽ báo lỗi ngay khi khởi tạo.

---

### `PolicyAuthenticityThresholds` ⭐⭐

```python
class PolicyAuthenticityThresholds(BaseModel):
    fabricated_policy_severity: IssueSeverity = CRITICAL  # Bịa = CRITICAL
    inaccurate_policy_severity: IssueSeverity = MAJOR     # Sai = MAJOR
    incomplete_policy_severity: IssueSeverity = MINOR     # Thiếu = MINOR
    missing_citation_severity: IssueSeverity = MAJOR      # Không có nguồn = MAJOR
    citation_required: bool = True                        # Phải có trích dẫn

    # Severity theo loại chính sách
    policy_type_severity: Dict[str, IssueSeverity] = {
        "warranty": CRITICAL,   # Bảo hành sai = nghiêm trọng nhất (rủi ro pháp lý)
        "return": MAJOR,        # Đổi trả sai = quan trọng
        "exchange": MAJOR,      # Đổi máy sai = quan trọng
        "service": MINOR,       # Sửa chữa sai = ít quan trọng hơn
        "support": MINOR,       # Hỗ trợ sai = ít quan trọng hơn
    }
```

**Tại sao warranty = CRITICAL nhưng support = MINOR?**

Nếu nói sai về bảo hành, khách có thể kiện vì đây là cam kết pháp lý. Nếu nói sai về giờ hỗ trợ, ít nghiêm trọng hơn nhiều.

**Method `classify_policy_issue()`:**

```python
def classify_policy_issue(self, is_fabricated, is_inaccurate, is_incomplete,
                           policy_type="service", has_citation=True) -> IssueSeverity:
    if is_fabricated:
        return IssueSeverity.CRITICAL  # Bịa đặt = luôn CRITICAL
    if self.citation_required and not has_citation:
        # Không có nguồn → severity theo loại chính sách
        type_severity = self.policy_type_severity.get(policy_type, MAJOR)
        if type_severity == CRITICAL:
            return CRITICAL
        return self.missing_citation_severity
    if is_inaccurate:
        return self.inaccurate_policy_severity  # MAJOR
    if is_incomplete:
        return self.incomplete_policy_severity  # MINOR
    return IssueSeverity.MINOR
```

---

### `TopicRelevanceThresholds` ⭐

```python
class TopicRelevanceThresholds(BaseModel):
    pass_coverage_threshold: float = 0.7    # >=70% → PASS
    minor_coverage_threshold: float = 0.8   # 50-80% → MINOR issue
    major_coverage_threshold: float = 0.5   # 30-50% → MAJOR issue
    critical_coverage_threshold: float = 0.3  # <30% → CRITICAL
    empathy_required: bool = True           # Phải có empathy
    min_empathy_score: float = 0.5          # Empathy score tối thiểu
    max_off_topic_ratio: float = 0.3        # Tối đa 30% nội dung lạc đề
```

---

### `EscalationThresholds` ⭐⭐

Quy tắc khi nào escalate cho người xử lý:

```python
class EscalationThresholds(BaseModel):
    max_critical_issues_before_escalation: int = 2  # >2 critical → escalate
    max_major_issues_before_escalation: int = 5     # >5 major → escalate
    max_total_issues_before_escalation: int = 10    # >10 issues → escalate

    # Escalate ngay lập tức (bỏ qua retry)
    fabricated_policy_immediate_escalation: bool = True  # Bịa chính sách → escalate ngay
    critical_price_deviation_escalation: bool = True     # Giá sai >30% → escalate ngay

    # Early termination
    early_termination_enabled: bool = True
    stop_on_first_critical: bool = False    # Mặc định: không dừng ngay critical đầu tiên
    multiple_critical_threshold: int = 3    # Dừng khi có 3+ critical

    # Số lần retry tùy theo severity
    max_retries_with_critical: int = 1    # Chỉ retry 1 lần nếu có critical
    max_retries_with_major: int = 3       # Retry 3 lần nếu chỉ có major
    max_retries_with_minor: int = 5       # Retry 5 lần nếu chỉ có minor
```

**Tại sao `stop_on_first_critical = False`?**

Mặc định, hệ thống không dừng ngay khi gặp critical đầu tiên. Lý do: muốn thu thập đủ thông tin về tất cả lỗi trước khi quyết định. Nếu bật `stop_on_first_critical = True`, sẽ dừng ngay → nhanh hơn nhưng có thể bỏ sót lỗi khác.

---

### `TimeoutConfig` ⭐

```python
class TimeoutConfig(BaseModel):
    llm_call: float = 10.0          # Timeout cho 1 LLM call (giây)
    price_check: float = 5.0        # Timeout cho price checker
    policy_check: float = 5.0       # Timeout cho policy checker
    relevance_check: float = 5.0    # Timeout cho relevance checker
    total_workflow: float = 30.0    # Timeout toàn bộ workflow
    escalate_on_critical_timeout: bool = True  # Timeout → escalate
```

**Method `get_timeout(operation)`:**
```python
def get_timeout(self, operation: str) -> float:
    return getattr(self, operation, self.llm_call)
    # Ví dụ: get_timeout("price_check") → 5.0
    # Ví dụ: get_timeout("unknown") → 10.0 (fallback về llm_call)
```

---

### `VerificationThresholdsConfig` — Top-level config ⭐

Gộp tất cả thresholds lại:

```python
class VerificationThresholdsConfig(BaseModel):
    price_accuracy: PriceAccuracyThresholds
    policy_authenticity: PolicyAuthenticityThresholds
    topic_relevance: TopicRelevanceThresholds
    escalation: EscalationThresholds
    timeouts: TimeoutConfig
    verification_weights: Dict[str, float] = {
        "price_accuracy": 0.4,      # Giá quan trọng nhất (40%)
        "policy_authenticity": 0.3, # Chính sách (30%)
        "topic_relevance": 0.3,     # Chủ đề (30%)
    }
```

**Environment overrides:**

```python
_ENVIRONMENT_OVERRIDES = {
    "development": {
        "price_accuracy": {"pass_tolerance_percent": 2.0},  # Lỏng hơn khi dev
        "escalation": {"early_termination_enabled": False},  # Tắt early termination
    },
    "production": {
        "price_accuracy": {"pass_tolerance_percent": 0.5},  # Chặt hơn khi production
        "escalation": {"max_critical_issues_before_escalation": 1},  # Escalate sớm hơn
    },
    "testing": {
        "escalation": {"early_termination_enabled": False},  # Tắt để test đầy đủ
    },
}
```

---

## 6. `verification/config/config_loader.py` ⭐

> File này xử lý việc **load và merge YAML config files**. Ít quan trọng hơn các file trên, nhưng cần hiểu để biết config được load như thế nào.

---

### `ConfigLoader` — Class chính

```python
class ConfigLoader:
    def __init__(self, base_path=None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._config_cache: Dict[str, Dict] = {}  # Cache để không load lại
```

**3 loại config có thể load:**

```python
loader.load_verification_config(environment)  # verification_config.yaml
loader.load_workflow_config(environment)       # workflow_config.yaml
loader.load_prompts_config(environment)        # prompts.yaml
```

**Deep merge logic — quan trọng:**

```python
def _merge_configs(self, base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = self._merge_configs(result[key], value)  # Merge đệ quy
        else:
            result[key] = deepcopy(value)  # Override hoàn toàn
    return result
```

Ví dụ:
```python
base = {"price": {"tolerance": 1.0, "critical": 30.0}}
override = {"price": {"tolerance": 0.5}}  # Chỉ override tolerance
result = {"price": {"tolerance": 0.5, "critical": 30.0}}  # critical vẫn giữ nguyên
```

Nếu không dùng deep merge, `critical` sẽ bị mất khi override.

**`reload()` — Reload không cần restart:**

```python
def reload(self, environment=None):
    self.clear_cache()  # Xóa cache cũ
    configs = self.load_all_configs(environment)  # Load lại
    self._emit_config_change("all")  # Thông báo cho các cache khác invalidate
    return configs
```

Sau khi reload, `_emit_config_change` sẽ invalidate price cache, policy cache, v.v. để chúng load lại với config mới.

---
## 7. `verification/agent/checkers.py` ⭐⭐⭐ QUAN TRỌNG

> File này chứa **3 checker thực hiện kiểm tra thực sự**. Đây là nơi logic nghiệp vụ quan trọng nhất nằm.

---

### Tổng quan: 3 checker làm gì?

```
PriceAccuracyChecker    → "Giá trong draft có đúng với database không?"
PolicyAuthenticityChecker → "Chính sách trong draft có thật không?"
TopicRelevanceChecker   → "Draft có trả lời đúng câu hỏi của khách không?"
```

Mỗi checker trả về `(bool, List[Issue])`:
- `(True, [])` → PASS, không có lỗi
- `(False, [issue1, issue2])` → FAIL, có danh sách lỗi

---

### `PriceAccuracyChecker` ⭐⭐⭐

**Khởi tạo — Dependency Injection:**

```python
class PriceAccuracyChecker:
    def __init__(self, llm, rag_pipeline, config, catalog_path, thresholds, price_cache):
        self.config = config
        self._thresholds = thresholds or PriceAccuracyThresholds(
            pass_tolerance_percent=config.price_tolerance_percent,
            critical_threshold_percent=config.price_critical_threshold,
        )
        self._early_termination = EarlyTerminationManager(thresholds_config)
        self._price_extractor = PriceExtractor()  # Extract giá từ text
        self._product_matcher = ProductMatcher(catalog_path)  # Fuzzy match sản phẩm
        self._price_cache = price_cache or get_product_price_cache()  # LRU cache
```

**`check_price_accuracy(draft, objection)` — Flow chính ⭐⭐⭐**

```python
def check_price_accuracy(self, draft: str, objection: str) -> Tuple[bool, List[PriceIssue]]:
    # Bước 1: Extract tất cả giá từ draft
    extracted_prices = self._price_extractor.extract(draft)
    # Ví dụ: ["iPhone 15 giá 35,000,000 VND", "Samsung S24 giá 25,000,000 VND"]

    # Bước 2: Nếu không có giá nhưng objection đề cập giá → FAIL
    if not extracted_prices:
        if self._objection_mentions_price(objection):
            return False, [PriceIssue(
                product_name="Unknown",
                severity=IssueSeverity.MAJOR,
                explanation="Objection mentions price but draft contains no pricing information"
            )]
        return True, []  # Không có giá, không cần kiểm tra

    # Bước 3: Kiểm tra từng giá
    issues = []
    overall_pass = True

    for extracted in extracted_prices:
        is_accurate, issue = self._verify_extracted_price(extracted)
        if not is_accurate and issue is not None:
            overall_pass = False
            issues.append(issue)

            # Bước 4: Check early termination sau mỗi lỗi
            termination = self._early_termination.should_terminate(issues)
            if termination.should_terminate:
                break  # Dừng ngay nếu gặp CRITICAL

    return overall_pass, issues
```

**`_verify_extracted_price(extracted)` — Kiểm tra một giá ⭐⭐⭐**

```python
def _verify_extracted_price(self, extracted: ExtractedPrice):
    # Bước 1: Tìm sản phẩm trong catalog
    match = self._find_product_from_context(extracted.product_context, extracted.amount_vnd)

    if match is None:
        return False, PriceIssue(
            product_name="Unknown Product",
            mentioned_price=extracted.original_text,
            severity=IssueSeverity.MAJOR,
            explanation="Cannot verify price — product not found in catalog"
        )

    # Bước 2: Tính deviation
    mentioned_vnd = extracted.amount_vnd
    actual_vnd = match.price_vnd
    deviation = abs(mentioned_vnd - actual_vnd) / actual_vnd * 100.0

    # Bước 3: Binary decision
    if self._thresholds.should_pass_price_check(deviation):
        return True, None  # PASS

    # Bước 4: Tạo PriceIssue với severity
    severity = self._thresholds.classify_price_deviation(deviation)
    return False, PriceIssue(
        product_name=match.display_name,
        product_sku=match.sku,
        mentioned_price=extracted.original_text,
        actual_price=match.price_raw,
        deviation_percent=round(deviation, 2),
        severity=severity,
        explanation=f"Price deviation {deviation:.1f}% exceeds tolerance {self._thresholds.pass_tolerance_percent}%",
        correction_suggestion=f"Update price to {match.price_raw} (SKU: {match.sku})"
    )
```

**`_find_product_from_context()` — Chiến lược tìm sản phẩm ⭐⭐**

Đây là phần thú vị nhất. Vấn đề: context có thể là "iPhone 15 Pro Max 256GB màu titan tự nhiên giá 35 triệu" — quá dài để fuzzy match chính xác.

```python
def _find_product_from_context(self, context: str, amount_vnd: float = 0.0):
    cache_key = f"{context}|{amount_vnd:.0f}"

    # Bước 1: Check cache trước (nhanh nhất)
    cached = self._price_cache.get(cache_key)
    if cached is not None:
        return cached  # Trả về ngay nếu đã có

    # Bước 2: Thử full context
    match = self._product_matcher.find_product(context)
    if match is not None:
        self._price_cache.put(cache_key, match)
        return match

    # Bước 3: Thử sub-strings ngắn hơn (2, 3, 4, 5 từ đầu)
    words = context.split()
    for n in [2, 3, 4, 5]:
        if n >= len(words):
            break
        sub_query = " ".join(words[:n])  # "iPhone 15", "iPhone 15 Pro", v.v.
        match = self._product_matcher.find_product(sub_query)
        if match is not None:
            self._price_cache.put(cache_key, match)
            return match

    # Bước 4: Thử với threshold thấp hơn (lenient hơn)
    original_threshold = self._product_matcher.threshold
    try:
        self._product_matcher.threshold = 0.4  # Giảm từ 0.7 xuống 0.4
        results = self._product_matcher.find_all(context, top_k=1)
        if results:
            self._price_cache.put(cache_key, results[0])
            return results[0]
    finally:
        self._product_matcher.threshold = original_threshold  # Restore

    # Bước 5: Không tìm thấy → cache None (negative cache)
    self._price_cache.put(cache_key, None)
    return None
```

**Tại sao cache cả `None`?**

Nếu không cache None, mỗi lần gặp sản phẩm không tìm thấy sẽ phải chạy lại toàn bộ 4 bước trên. Cache None (negative cache) giúp tránh điều này.

---

### `PolicyAuthenticityChecker` ⭐⭐⭐

**`_POLICY_TAXONOMY` — Dictionary định nghĩa keywords:**

```python
_POLICY_TAXONOMY = {
    'warranty': {
        'keywords': ['bảo hành', 'warranty', 'guarantee', ...],
        'duration_patterns': [
            r'(?:bảo hành|warranty)\s+(\d+)\s*(năm|tháng|year|month)',
        ],
        'claim_patterns': [
            r'(?:bảo hành|warranty)\s+(?:toàn|full|complete)',
            r'(?:miễn phí|free)\s+(?:bảo hành|warranty)',
        ],
    },
    'return': { ... },
    'exchange': { ... },
    'service': { ... },
    'support': { ... },
}
```

**`_extract_policy_statements(text)` — Tìm tất cả câu đề cập chính sách ⭐⭐**

```python
def _extract_policy_statements(self, text: str):
    # Bước 1: Split text thành sentences
    sentences = self._split_into_sentences(text)

    raw_matches = []
    for policy_type, taxonomy in self._POLICY_TAXONOMY.items():
        for keyword in taxonomy['keywords']:
            for sentence, (start, end) in sentences:
                if keyword not in sentence.lower():
                    continue

                # Bước 2: Extract duration claim
                # Ví dụ: "bảo hành 12 tháng" → {"amount": "12", "unit": "tháng"}
                duration = self._extract_duration(sentence, taxonomy['duration_patterns'])

                # Bước 3: Extract specific claims
                # Ví dụ: "bảo hành miễn phí" → ["bảo hành miễn phí"]
                claims = self._extract_claims(sentence, taxonomy['claim_patterns'])

                # Bước 4: Tính confidence score
                confidence = 0.5                    # Base score
                if ' ' in keyword: confidence += 0.2  # Multi-word keyword → chính xác hơn
                confidence += min(0.3, len(claims) * 0.15)  # Có claims → chính xác hơn
                if duration: confidence += 0.1      # Có duration → chính xác hơn

                raw_matches.append({
                    'text': sentence,
                    'type': policy_type,
                    'keyword': keyword,
                    'duration': duration,
                    'claims': claims,
                    'confidence': confidence,
                })

    # Bước 5: Deduplicate — nếu 2 matches cùng sentence, giữ cái confidence cao hơn
    return self._deduplicate_statements(raw_matches)
```

**`_verify_policy_statement(statement)` — Kiểm tra một câu chính sách ⭐⭐⭐**

```python
def _verify_policy_statement(self, statement):
    policy_type = statement.get('type', 'service')

    # Bước 1: Check forbidden phrases → CRITICAL ngay
    if self._contains_forbidden_phrases(statement['text']):
        return False, PolicyIssue(
            is_fabricated=True,
            severity=IssueSeverity.CRITICAL,
            explanation="Policy contains forbidden phrases indicating fabrication"
        )

    # Bước 2: Lookup trong DB qua RAG pipeline
    is_verified, correct_policy = self._lookup_policy_in_db(statement)

    if is_verified:
        return True, None  # PASS

    # Bước 3: Phân loại lỗi
    is_fabricated = correct_policy is None   # Không tìm thấy gì → bịa đặt
    is_inaccurate = correct_policy is not None  # Tìm thấy nhưng không khớp → sai

    severity = self._thresholds.classify_policy_issue(
        is_fabricated=is_fabricated,
        is_inaccurate=is_inaccurate,
        policy_type=policy_type,
    )

    return False, PolicyIssue(
        is_fabricated=is_fabricated,
        is_inaccurate=is_inaccurate,
        correct_policy=correct_policy,
        severity=severity,
        explanation="Policy statement not found in official documents" if is_fabricated
                    else "Policy statement inaccurate compared to official documents"
    )
```

**`_lookup_policy_in_db(statement)` — Tra cứu trong tài liệu chính thức:**

```python
def _lookup_policy_in_db(self, statement):
    # Graceful fallback nếu không có RAG pipeline
    if self.rag_pipeline is None:
        return True, None  # Không thể verify → assume PASS

    # Bước 1: Build targeted query
    base_query = self._POLICY_QUERY_TEMPLATES.get(policy_type, keyword)
    # Ví dụ: "chính sách bảo hành warranty policy terms"

    # Bước 2: Retrieve top-K chunks từ RAG
    chunks = self.rag_pipeline.retriever.retrieve(query, top_k=5)

    # Bước 3: Filter chunks là policy documents
    policy_chunks = [c for c in chunks if "policy" in c.source_type]

    # Bước 4: So sánh claims với retrieved text
    for claim in statement['claims']:
        if not any(claim in chunk.text for chunk in policy_chunks):
            return False, policy_chunks[0].text if policy_chunks else None

    return True, None  # Tất cả claims đều verified
```

---

### `TopicRelevanceChecker` ⭐⭐

Checker này dùng `SemanticSimilarityAnalyzer` và `IntentClassifier`:

```python
class TopicRelevanceChecker:
    def __init__(self, llm, config):
        self._semantic_analyzer = SemanticSimilarityAnalyzer()
        self._intent_classifier = IntentClassifier()

    def check_topic_relevance(self, draft: str, objection: str):
        # Bước 1: Phân tích intent của objection
        intents = self._intent_classifier.classify(objection)
        # Ví dụ: ["price_comparison", "value_justification", "feature_inquiry"]

        # Bước 2: Tính coverage ratio
        coverage = self._calculate_coverage(intents, draft)
        # Ví dụ: 0.6 = chỉ trả lời được 60% intents

        # Bước 3: Tính empathy score
        empathy_score = self._calculate_empathy(draft)

        # Bước 4: Binary decision
        if coverage >= self._thresholds.pass_coverage_threshold:
            return True, []  # PASS

        severity = self._thresholds.classify_relevance_issue(coverage)
        return False, [RelevanceIssue(
            objection_intent=str(intents),
            response_coverage=coverage,
            missing_aspects=self._find_missing_aspects(intents, draft),
            empathy_score=empathy_score,
            severity=severity,
        )]
```

---
## 8. `verification/agent/verification_agent.py` ⭐⭐⭐ QUAN TRỌNG

> File này chứa `VerificationAgent` — **orchestrator chạy 3 checker song song**, quản lý cache, circuit breaker, và error handling. Đây là trái tim của verification layer.

---

### `CircuitBreaker` — Pattern bảo vệ external services ⭐⭐⭐

Trước khi hiểu `VerificationAgent`, cần hiểu `CircuitBreaker`.

**Vấn đề:** LLM API, RAG pipeline, database đôi khi bị lỗi. Nếu cứ gọi liên tục khi service đang lỗi → tốn thời gian chờ timeout, tốn tài nguyên.

**Giải pháp:** Circuit Breaker — giống như cầu dao điện. Khi phát hiện lỗi liên tục, "ngắt mạch" để không gọi nữa, dùng fallback thay thế.

```
CLOSED (bình thường)
    ↓ 3 lần lỗi liên tiếp
OPEN (ngắt mạch — block requests)
    ↓ sau 60 giây
HALF_OPEN (thử lại 1 request)
    ↓ thành công          ↓ thất bại
CLOSED (phục hồi)      OPEN (ngắt lại)
```

**Code thực tế:**

```python
class CircuitBreaker:
    def __init__(self, service_name, failure_threshold=3, recovery_timeout_seconds=60.0):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._lock = threading.Lock()  # Thread-safe

    def allow_request(self) -> bool:
        with self._lock:
            state = self._get_state()
            if state == CircuitBreakerState.CLOSED:
                return True   # Bình thường → cho qua
            if state == CircuitBreakerState.HALF_OPEN:
                return True   # Đang test → cho 1 request qua
            return False      # OPEN → block

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED  # Phục hồi

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN  # Ngắt mạch
```

**Tại sao dùng `threading.Lock()`?**

Nhiều requests có thể chạy đồng thời (async). Nếu không có lock, 2 threads có thể cùng đọc `_failure_count = 2`, cùng tăng lên 3, cùng set OPEN → race condition. Lock đảm bảo chỉ 1 thread thay đổi state tại một thời điểm.

---

### `VerificationAgent` — Class chính ⭐⭐⭐

**Khởi tạo:**

```python
class VerificationAgent:
    def __init__(self, llm, rag_pipeline, config, max_concurrent_llm_calls=10, compression_level="none"):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.config = config

        # Semaphore giới hạn concurrent LLM calls
        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        # Nếu có 10 requests đồng thời, chỉ 10 cái được gọi LLM cùng lúc
        # Cái thứ 11 phải chờ

        # 3 checker
        self.price_checker = PriceAccuracyChecker(llm, rag_pipeline, config)
        self.policy_checker = PolicyAuthenticityChecker(llm, rag_pipeline, config)
        self.relevance_checker = TopicRelevanceChecker(llm, config)

        # LRU cache cho kết quả verification
        if config.enable_caching:
            self._cache = VerificationCache(
                max_size=config.cache_max_size,
                default_ttl_seconds=float(config.cache_ttl_seconds),
            )

        # Circuit breakers cho từng service
        self._circuit_breakers = {
            "llm_api": CircuitBreaker("llm_api", failure_threshold=3, recovery_timeout_seconds=60.0),
            "rag_pipeline": CircuitBreaker("rag_pipeline", failure_threshold=3, recovery_timeout_seconds=30.0),
            "db": CircuitBreaker("db", failure_threshold=3, recovery_timeout_seconds=30.0),
        }

        # Lưu kết quả cuối cùng hợp lệ (dùng khi DB lỗi)
        self._last_valid_result = None
```

---

### `verify_draft(state)` — Entry point chính ⭐⭐⭐

```python
async def verify_draft(self, state: WorkflowState) -> VerificationResult:
    start_time = time.time()
    correlation_id = self._get_correlation_id()

    async with self._semaphore:  # Giới hạn concurrent calls
        try:
            # Bước 1: Validate input
            self._validate_input(state)

            # Bước 2: Check cache
            if self._cache is not None:
                cached = self._cache.get(self._generate_cache_key(state))
                if cached is not None:
                    return cached  # Trả về ngay nếu đã có

            # Bước 3: Chạy verification
            if self.config.parallel_verification:
                verification_result = await self._verify_parallel(state)
            else:
                verification_result = await self._verify_sequential(state)

            # Bước 4: Set execution time
            execution_time = time.time() - start_time
            verification_result.execution_time_seconds = execution_time

            # Bước 5: Cache kết quả
            if self._cache is not None:
                self._cache.put(self._generate_cache_key(state), verification_result)

            # Bước 6: Lưu làm last valid result (cho DB fallback)
            self._last_valid_result = verification_result

            return verification_result

        except Exception as e:
            execution_time = time.time() - start_time
            return self._handle_verification_error(state, e, execution_time)
```

**Tại sao dùng `async with self._semaphore`?**

Nếu 100 requests đến cùng lúc và tất cả đều gọi LLM → LLM API bị quá tải, rate limit, hoặc tốn tiền quá nhiều. Semaphore giới hạn chỉ 10 requests được gọi LLM cùng lúc, 90 cái còn lại phải chờ.

---

### `_verify_parallel(state)` — Parallel với first-failure-fast ⭐⭐⭐

Đây là phần kỹ thuật quan trọng nhất trong file này.

**Vấn đề:** Chạy 3 checker tuần tự mất 3x thời gian. Chạy song song mất 1x thời gian. Nhưng nếu checker đầu tiên phát hiện CRITICAL, không cần chạy 2 checker còn lại.

**Giải pháp: First-failure-fast với `asyncio.wait(FIRST_COMPLETED)`**

```python
async def _verify_parallel(self, state: WorkflowState) -> VerificationResult:
    if not self.config.early_termination:
        return await self._verify_parallel_simple(state)  # Chạy đơn giản

    tracker = AsyncStepLatencyTracker()  # Track latency từng checker

    # Tạo 3 tasks chạy đồng thời
    check_map = {
        "price": asyncio.ensure_future(self._check_price_accuracy_async(state)),
        "policy": asyncio.ensure_future(self._check_policy_authenticity_async(state)),
        "relevance": asyncio.ensure_future(self._check_topic_relevance_async(state)),
    }
    task_to_type = {v: k for k, v in check_map.items()}

    results = {}
    pending = set(check_map.values())

    while pending:
        # Chờ task nào xong trước
        done, pending = await asyncio.wait(
            pending,
            timeout=self.config.async_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout → cancel tất cả
            for t in pending:
                t.cancel()
            raise RuntimeError(f"Verification timeout after {self.config.async_timeout_seconds}s")

        for task in done:
            check_type = task_to_type[task]
            if task.exception():
                results[check_type] = task.exception()
            else:
                results[check_type] = task.result()

            # Kiểm tra early termination
            result = results[check_type]
            if not isinstance(result, Exception):
                _, issues = result
                if self._has_critical_issues(issues):
                    # CRITICAL found! Cancel remaining tasks
                    for t in pending:
                        t.cancel()
                    # Fill missing với default PASS (không phải FAIL)
                    for remaining in [t for t in task_to_type.values() if t not in results]:
                        results[remaining] = (True, [])
                    pending = set()
                    break

    # Unpack results
    price_pass, price_issues = self._unpack(results.get("price", (True, [])))
    policy_pass, policy_issues = self._unpack(results.get("policy", (True, [])))
    relevance_pass, relevance_issues = self._unpack(results.get("relevance", (True, [])))

    return self._build_verification_result(
        state, price_pass, price_issues, policy_pass, policy_issues, relevance_pass, relevance_issues
    )
```

**Tại sao fill missing với `(True, [])` (PASS) thay vì FAIL?**

Nếu price checker phát hiện CRITICAL và cancel policy/relevance checker, chúng ta không biết policy và relevance có lỗi không. Nếu fill với FAIL → có thể báo lỗi sai (false positive). Fill với PASS → chỉ báo lỗi những gì đã kiểm tra được.

**Ví dụ minh họa:**

```
Thời gian:  0ms    500ms   1000ms  1500ms  2000ms
price:      ──────────────── CRITICAL! ──────────────
policy:     ──────────────────────────────── (cancelled)
relevance:  ──────────────────────────────── (cancelled)

Kết quả: price=FAIL(CRITICAL), policy=PASS(default), relevance=PASS(default)
→ Tiết kiệm ~1000ms và 2 LLM calls
```

---

### `verify_draft_with_degradation(state)` — Graceful degradation ⭐⭐

Khi 1 checker bị lỗi (exception), không nên fail toàn bộ verification:

```python
async def verify_draft_with_degradation(self, state):
    handler = GracefulDegradationHandler()

    # Chạy 3 checkers, bắt exception thay vì raise
    price_pr, policy_pr, relevance_pr = await asyncio.gather(
        handler.run_checker_safely("price", self.price_checker.check_price_accuracy, ...),
        handler.run_checker_safely("policy", self.policy_checker.check_policy_authenticity, ...),
        handler.run_checker_safely("relevance", self.relevance_checker.check_topic_relevance, ...),
    )

    failed_checkers = [name for name, pr in results.items() if not pr.success]

    if len(failed_checkers) >= 2:
        # 2+ checkers fail → không thể tiếp tục
        return self._build_fallback_result(...)

    # 1 checker fail → tiếp tục với 2 checker còn lại + warning
    return handler.aggregate_partial_results(partial_results, reasoning)
```

**Khi nào dùng `verify_draft_with_degradation` thay vì `verify_draft`?**

- `verify_draft`: Môi trường ổn định, muốn kết quả chính xác nhất
- `verify_draft_with_degradation`: Môi trường không ổn định, muốn hệ thống tiếp tục chạy dù có lỗi

---

### `verify_draft_sync(state)` — Sync wrapper ⭐

LangGraph node là sync function, nhưng `verify_draft` là async. Cần wrapper:

```python
def verify_draft_sync(self, state: WorkflowState) -> VerificationResult:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Đang trong async context (pytest-asyncio, v.v.)
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(self.verify_draft(state), loop)
            return future.result(timeout=self.config.async_timeout_seconds)
        else:
            return loop.run_until_complete(self.verify_draft(state))
    except RuntimeError:
        # Không có event loop → tạo mới
        return asyncio.run(self.verify_draft(state))
```

---
## 9. `verification/workflow/routing.py` ⭐⭐⭐ QUAN TRỌNG

> File này chứa **logic routing thuần Python** — quyết định workflow đi đâu tiếp theo sau mỗi bước. Đây là "bộ não" điều phối của workflow.

---

### Tại sao routing là pure Python, không dùng AI? ⭐⭐⭐

Đây là quyết định thiết kế quan trọng. Routing có thể dùng LLM để quyết định "có nên escalate không?", nhưng hệ thống chọn pure Python vì:

| | Pure Python | LLM |
|--|-------------|-----|
| Tốc độ | Microseconds | Seconds |
| Chi phí | Miễn phí | Tốn token |
| Nhất quán | Cùng input → cùng output | Có thể khác nhau |
| Debug | Dễ trace | Khó hiểu tại sao |
| Test | Dễ unit test | Khó test |

Routing là logic đơn giản (if/else), không cần AI.

---

### `WorkflowRouter` — Class chính ⭐⭐⭐

**`route_after_verification(state)` — Quyết định sau verification node:**

```python
def route_after_verification(self, state: WorkflowState) -> Literal["approved", "correction", "escalation"]:
    verification_result = state.get("verification_result")

    # Case 1: Không có kết quả → lỗi kỹ thuật → escalate
    if verification_result is None:
        return "escalation"

    # Case 2: PASS → kết thúc thành công
    if verification_result.is_approved:
        return "approved"

    # Case 3: immediate_termination flag được set bởi CriticalIssueDetector
    if getattr(verification_result, "immediate_termination", False):
        return "escalation"

    # Case 4: Kiểm tra các điều kiện escalate ngay
    if self._should_escalate_immediately(verification_result, state):
        return "escalation"

    # Case 5: Đã hết retry
    if state["retry_count"] >= state["max_retries"]:
        return "escalation"

    # Case 6: Còn retry → sửa lại
    return "correction"
```

**Thứ tự kiểm tra quan trọng:** Kiểm tra từ nghiêm trọng nhất đến ít nghiêm trọng nhất. Nếu đảo thứ tự, có thể bỏ sót điều kiện quan trọng.

---

### `_should_escalate_immediately()` — Khi nào escalate ngay? ⭐⭐

```python
def _should_escalate_immediately(self, verification_result, state) -> bool:
    # Điều kiện 1: CriticalIssueDetector phát hiện critical
    if getattr(self.config, 'critical_issue_escalation', True):
        termination_decision = self._critical_detector.check_verification_result(verification_result)
        if termination_decision.should_terminate:
            return True

    # Điều kiện 2: Quá nhiều issues (systemic problem)
    total_issues = self._count_total_issues(verification_result)
    if total_issues >= 5:
        return True

    # Điều kiện 3: Có chính sách bịa đặt (rủi ro pháp lý)
    fabricated_policies = sum(
        1 for issue in verification_result.criteria.policy_issues
        if issue.is_fabricated and issue.severity == IssueSeverity.CRITICAL
    )
    if fabricated_policies > 0:
        return True

    return False
```

---

### `route_after_correction(state)` — Quyết định sau correction node ⭐⭐

```python
def route_after_correction(self, state) -> Literal["retry", "escalation"]:
    # Điều kiện 1: Đã hết retry
    if state["retry_count"] >= state["max_retries"]:
        return "escalation"

    # Điều kiện 2: Đã retry nhiều lần mà vẫn có critical issues
    verification_result = state.get("verification_result")
    if verification_result and self._should_escalate_after_correction(verification_result, state):
        return "escalation"

    # Còn lại → retry
    return "retry"
```

---

### `_is_complex_issue_pattern()` — Phát hiện patterns phức tạp ⭐⭐

```python
def _is_complex_issue_pattern(self, verification_result) -> bool:
    # Pattern 1: Tất cả 3 tiêu chí đều fail → systemic problem
    failed_count = sum([
        not verification_result.criteria.price_accuracy_pass,
        not verification_result.criteria.policy_authenticity_pass,
        not verification_result.criteria.topic_relevance_pass
    ])
    if failed_count >= 3:
        return True

    # Pattern 2: Critical price + critical policy + major relevance cùng lúc
    has_critical_price = any(i.severity == CRITICAL for i in verification_result.criteria.price_issues)
    has_critical_policy = any(i.severity == CRITICAL for i in verification_result.criteria.policy_issues)
    has_major_relevance = any(i.severity == MAJOR for i in verification_result.criteria.relevance_issues)
    if sum([has_critical_price, has_critical_policy, has_major_relevance]) >= 2:
        return True

    # Pattern 3: Price deviation > 50% ở 2+ sản phẩm
    high_deviations = sum(
        1 for issue in verification_result.criteria.price_issues
        if issue.deviation_percent and issue.deviation_percent > 50.0
    )
    if high_deviations >= 2:
        return True

    # Pattern 4: 2+ fabricated policies
    fabricated_count = sum(1 for issue in verification_result.criteria.policy_issues if issue.is_fabricated)
    if fabricated_count >= 2:
        return True

    # Pattern 5: Response coverage < 30%
    low_coverage = sum(
        1 for issue in verification_result.criteria.relevance_issues
        if issue.response_coverage < 0.3
    )
    if low_coverage > 0:
        return True

    return False
```

**Tại sao cần detect complex patterns?**

Nếu Sales Agent liên tục tạo ra draft với nhiều lỗi nghiêm trọng, retry sẽ không giúp ích. Cần người xử lý để điều tra nguyên nhân gốc rễ (có thể là dữ liệu training sai, hoặc prompt không đúng).

---

## 10. `verification/workflow/correction.py` ⭐⭐⭐ QUAN TRỌNG

> File này tạo **structured correction feedback** — đoạn text hướng dẫn Sales Agent sửa lỗi khi retry. Chất lượng của feedback này quyết định Sales Agent có sửa đúng không.

---

### `SelfCorrectionNode` — Class chính ⭐⭐⭐

**`generate_correction_feedback()` — Method quan trọng nhất:**

```python
def generate_correction_feedback(self, original_objection, failed_draft,
                                  verification_result, feedback_report=None) -> str:
    if verification_result.is_approved:
        return "✅ No corrections needed - verification passed"

    # Nếu có FeedbackReport sẵn → dùng luôn (không cần build lại)
    if feedback_report is not None and not feedback_report.is_approved:
        return "\n\n".join([
            feedback_report.correction_prompt,
            self._build_retry_instructions_section(original_objection, failed_draft),
            self._build_quality_checklist_section(),
        ])

    # Không có FeedbackReport → build từ đầu với 5 sections
    return "\n\n".join([
        self._build_header_section(verification_result),
        self._build_issue_analysis_section(verification_result),
        self._build_specific_corrections_section(verification_result),
        self._build_retry_instructions_section(original_objection, failed_draft),
        self._build_quality_checklist_section(),
    ])
```

**5 sections của correction feedback:**

**Section 1 — Header (Tóm tắt vấn đề):**
```
🔄 VERIFICATION FAILED - CORRECTION REQUIRED
==================================================
📊 Issue Summary: 2 total issues detected
⚠️  CRITICAL: 0 critical issues require immediate attention
❌ Failed Criteria:
  • Price Accuracy: ❌ FAILED
  • Policy Authenticity: ✅ PASSED
  • Topic Relevance: ❌ FAILED
```

**Section 2 — Issue Analysis (Chi tiết từng lỗi):**
```
📋 DETAILED ISSUE ANALYSIS:

💰 PRICE ACCURACY ISSUES:
  1. ⚠️ iPhone 15 Pro Max
     • Mentioned: 35,000,000 VND
     • Actual: 29,990,000 VND
     • Deviation: 16.7%
     • Issue: Price deviation 16.7% exceeds tolerance 1%

🎯 TOPIC RELEVANCE ISSUES:
  1. ⚠️ Response Coverage: 60.0%
     • Objection Intent: So sánh giá iPhone vs Samsung
     • Missing Aspects: camera comparison, gaming performance
```

**Section 3 — Specific Corrections (Hướng dẫn sửa cụ thể):**
```
🛠️  SPECIFIC CORRECTIONS REQUIRED:

💰 Price Accuracy Corrections:
  • Update iPhone 15 Pro Max price from '35,000,000 VND' to '29,990,000 VND'

🎯 Topic Relevance Corrections:
  • Address missing aspects: camera comparison, gaming performance
  • Significantly expand response to better address objection
```

**Section 4 — Retry Instructions (Hướng dẫn chung):**
```
🔄 RETRY INSTRUCTIONS:

When generating the corrected response, you MUST:
1. 📊 Address ALL issues listed above in priority order (Critical → Major → Minor)
2. 🔍 Cross-check ALL price information against the internal database
3. 📋 Verify ALL policy statements against official documents
4. 🎯 Ensure response directly addresses the customer's specific objection

📝 ORIGINAL OBJECTION TO ADDRESS:
"iPhone quá đắt so với Samsung, tại sao tôi nên mua?"
```

**Section 5 — Quality Checklist (Checklist tự kiểm tra):**
```
✅ QUALITY CHECKLIST - Verify before submitting:

Price Accuracy:
  □ All prices match internal database exactly
  □ Product names and SKUs are correct

Policy Authenticity:
  □ All policies quoted from official documents
  □ No fabricated or assumed policy statements

Topic Relevance:
  □ Response directly addresses customer objection
  □ All objection components are covered
```

**Tại sao cần 5 sections?**

- **Header**: Sales Agent biết ngay có bao nhiêu lỗi, loại gì
- **Issue Analysis**: Biết chính xác lỗi ở đâu, sai bao nhiêu
- **Specific Corrections**: Biết chính xác cần làm gì
- **Retry Instructions**: Nhắc nhở các nguyên tắc chung
- **Quality Checklist**: Tự kiểm tra trước khi submit

Nếu chỉ có "giá sai" mà không có "sửa thành bao nhiêu", Sales Agent vẫn có thể sai lần nữa.

---
## 11. `verification/workflow/workflow.py` ⭐⭐⭐ QUAN TRỌNG NHẤT

> File này là **orchestrator chính** — nơi tất cả các thành phần được kết nối lại với nhau thành một workflow hoàn chỉnh. Đây là entry point của toàn bộ hệ thống.

---

### `VerificationWorkflow` — Class chính ⭐⭐⭐

**Khởi tạo:**

```python
class VerificationWorkflow:
    def __init__(self, research_agent, verification_agent, config, persistence_manager=None):
        self.research_agent = research_agent      # Sales Research Agent
        self.verification_agent = verification_agent  # Verification Agent
        self.config = config

        # Các component workflow
        self.correction_node = SelfCorrectionNode(config)
        self.router = WorkflowRouter(config)

        # Persistence manager (optional) — để checkpoint/resume
        self.persistence_manager = persistence_manager

        # Build StateGraph
        self.graph = self._build_graph()
```

---

### `_build_graph()` — Xây dựng StateGraph ⭐⭐⭐

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(WorkflowState)

    # Đăng ký 4 nodes
    workflow.add_node("research", self._execute_research_node)
    workflow.add_node("verification", self._execute_verification_node)
    workflow.add_node("correction", self._execute_correction_node)
    workflow.add_node("escalation", self._execute_escalation_node)

    # Entry point: bắt đầu từ research
    workflow.set_entry_point("research")

    # Edge cố định: research → verification (luôn luôn)
    workflow.add_edge("research", "verification")

    # Edge có điều kiện: verification → ?
    workflow.add_conditional_edges(
        "verification",
        self._route_after_verification,  # Hàm routing
        {
            "approved": END,              # PASS → kết thúc
            "correction": "correction",   # FAIL → sửa
            "escalation": "escalation"    # CRITICAL → escalate
        }
    )

    # Edge có điều kiện: correction → ?
    workflow.add_conditional_edges(
        "correction",
        self._route_after_correction,
        {
            "retry": "research",          # Retry → quay lại research
            "escalation": "escalation"    # Quá nhiều retry → escalate
        }
    )

    # Edge cố định: escalation → END (luôn luôn)
    workflow.add_edge("escalation", END)

    return workflow.compile()
```

**Visualize đồ thị:**

```
START
  ↓
[research] ──────────────────────────────────────────────────────────────────┐
  ↓                                                                           │
[verification]                                                                │
  ↓                                                                           │
  ├── is_approved=True ──────────────────────────────────────────────── END ✅│
  │                                                                           │
  ├── FAIL (có thể retry) ──→ [correction] ──→ retry ──────────────────────┘
  │                                  ↓
  │                             escalation
  │                                  ↓
  └── CRITICAL / max_retries ──→ [escalation] ──→ END 🚨
```

---

### `execute_workflow()` — Entry point public ⭐⭐⭐

```python
async def execute_workflow(self, objection_text: str, customer_context=None) -> WorkflowState:
    # Bước 1: Tạo workflow_id unique
    workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Bước 2: Check persistence — nếu có checkpoint thì resume
    if self.persistence_manager and self.persistence_manager.exists(workflow_id):
        initial_state = self.persistence_manager.load(workflow_id)
    else:
        initial_state = self._build_initial_state(workflow_id, objection_text, customer_context)

    # Bước 3: Set correlation context cho distributed tracing
    correlation_id = initial_state.get("correlation_id")
    set_correlation_context(correlation_id, workflow_id)

    # Bước 4: Register với shutdown manager (graceful shutdown)
    current_task = asyncio.current_task()
    if current_task:
        get_shutdown_manager().register_task(current_task)

    tracer = get_tracer()
    try:
        # Bước 5: Execute với distributed tracing span
        async with tracer.start_workflow_span(workflow_id, correlation_id):
            final_state = await self._execute_graph_async(initial_state)
            final_state["end_time"] = datetime.now().isoformat()

            # Bước 6: Attach trace data vào execution log
            trace_data = tracer.export_trace(workflow_id)
            if trace_data and final_state["execution_log"]:
                final_state["execution_log"][-1].metrics["trace"] = trace_data

        # Bước 7: Delete checkpoint nếu thành công
        if self.persistence_manager:
            self.persistence_manager.delete(workflow_id)

        return final_state

    except Exception as e:
        return self._handle_workflow_error(initial_state, e)
        # Checkpoint được giữ lại để resume sau

    finally:
        tracer.clear_trace(workflow_id)
        clear_correlation_context()
```

---

### `_execute_research_node(state)` — Research node ⭐⭐⭐

```python
def _execute_research_node(self, state: WorkflowState) -> WorkflowState:
    start_time = datetime.now()

    try:
        state["workflow_status"] = "researching"

        # Xây dựng prompt
        objection = state["objection_text"]
        correction_feedback = state.get("correction_feedback")

        if correction_feedback and state.get("retry_count", 0) > 0:
            # Lần retry: prepend correction feedback vào prompt
            prompt = f"{correction_feedback}\n\n---\nORIGINAL OBJECTION:\n{objection}"
        else:
            # Lần đầu: chỉ có objection
            prompt = objection

        # Dispatch đến research agent (hỗ trợ 3 interface khác nhau)
        if hasattr(self.research_agent, "run"):
            agent_result = self.research_agent.run(prompt)
            state["draft_response"] = agent_result.draft_response
            state["tools_used"] = agent_result.tools_used
        elif hasattr(self.research_agent, "process_objection"):
            result = self.research_agent.process_objection(objection, correction_feedback=correction_feedback)
            state["draft_response"] = result.get("response", "")
        else:
            state["draft_response"] = self.research_agent.generate_response(prompt)

        # Tạo ExecutionStep
        execution_time = (datetime.now() - start_time).total_seconds()
        execution_step = ExecutionStep(
            node_name="research",
            execution_time=execution_time,
            status=ExecutionStatus.SUCCESS,
            input_summary=f"objection: '{objection[:97]}...'",
            output_summary=f"draft: '{state['draft_response'][:97]}...'",
            metrics={"retry_count": state.get("retry_count", 0), "tools_used": state["tools_used"]},
        )
        state["execution_log"].append(execution_step)

        # Auto-checkpoint
        self._checkpoint(state)
        return state

    except Exception as e:
        return self._handle_node_error(state, "research", e, start_time)
```

**Tại sao hỗ trợ 3 interface?**

Để không bị lock-in vào một implementation cụ thể. Nếu Sales Agent được refactor, workflow vẫn hoạt động miễn là có một trong 3 interface.

---

### `_execute_verification_node(state)` — Verification node ⭐⭐⭐

```python
def _execute_verification_node(self, state: WorkflowState) -> WorkflowState:
    start_time = datetime.now()

    try:
        state["workflow_status"] = "verifying"

        # Handle event loop (LangGraph là sync, verification_agent là async)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Đang trong async context → dùng thread-safe future
            future = asyncio.run_coroutine_threadsafe(
                self.verification_agent.verify_draft(state), loop
            )
            verification_result = future.result(timeout=self.config.async_timeout_seconds)
        else:
            verification_result = loop.run_until_complete(
                self.verification_agent.verify_draft(state)
            )

        # Update state
        state["verification_result"] = verification_result

        if verification_result.is_approved:
            state["workflow_status"] = "approved"
            state["final_response"] = state.get("draft_response", "")
        else:
            state["workflow_status"] = "correction_needed"

        # Fire critical alerts (fire-and-forget — không block workflow)
        try:
            alert_manager = get_critical_alert_manager()
            alert_manager.check_and_alert(verification_result, ...)
        except Exception:
            pass  # Alert fail không được crash workflow

        # Tạo ExecutionStep với đầy đủ metrics
        execution_step = ExecutionStep(
            node_name="verification",
            execution_time=(datetime.now() - start_time).total_seconds(),
            status=ExecutionStatus.SUCCESS,
            output_summary=f"result: {'APPROVED' if verification_result.is_approved else 'FAILED'}",
            metrics={
                "overall_pass": verification_result.criteria.overall_pass,
                "critical_issues": verification_result.criteria.critical_issues_count,
                "price_accuracy_pass": verification_result.criteria.price_accuracy_pass,
                "policy_authenticity_pass": verification_result.criteria.policy_authenticity_pass,
                "topic_relevance_pass": verification_result.criteria.topic_relevance_pass,
                "tokens_used": verification_result.llm_tokens_used,
                **verification_result.step_latencies or {},
            },
        )
        state["execution_log"].append(execution_step)

        self._checkpoint(state)
        return state

    except Exception as e:
        return self._handle_node_error(state, "verification", e, start_time)
```

**Tại sao phải handle event loop phức tạp như vậy?**

LangGraph `graph.invoke()` là sync. Nhưng `verify_draft()` là async. Khi gọi async từ sync, cần event loop. Vấn đề là:
- Nếu đang trong pytest-asyncio → loop đang chạy → không thể `run_until_complete`
- Nếu không có loop → cần tạo mới

Code trên xử lý cả 2 trường hợp.

---

### `_checkpoint(state)` — Auto-checkpoint ⭐

```python
def _checkpoint(self, state: WorkflowState) -> None:
    if self.persistence_manager is None:
        return  # Không có persistence → skip

    try:
        self.persistence_manager.save(state)
    except Exception as exc:
        logger.warning("Auto-checkpoint failed: %s", exc)
        # KHÔNG raise exception — checkpoint fail không được crash workflow
```

**Tại sao checkpoint quan trọng?**

Nếu server crash sau khi research node chạy xong nhưng trước khi verification node chạy, không có checkpoint → phải chạy lại từ đầu (tốn tiền gọi LLM lại). Với checkpoint, có thể resume từ điểm đã dừng.

---

### `_execute_graph_async(initial_state)` — Chạy graph async ⭐

```python
async def _execute_graph_async(self, initial_state: WorkflowState) -> WorkflowState:
    loop = asyncio.get_event_loop()

    def run_graph():
        return self.graph.invoke(initial_state)  # Sync call

    # Wrap sync call trong executor để không block event loop
    try:
        final_state = await asyncio.wait_for(
            loop.run_in_executor(None, run_graph),
            timeout=self.config.async_timeout_seconds * 3  # 3x timeout cho toàn workflow
        )
        return final_state
    except asyncio.TimeoutError:
        raise RuntimeError(f"Workflow timeout after {self.config.async_timeout_seconds * 3}s")
```

**Tại sao timeout = `async_timeout_seconds * 3`?**

`async_timeout_seconds` là timeout cho một verification (30s). Toàn bộ workflow có thể có 3 lần retry, mỗi lần có research + verification → cần ít nhất 3x thời gian.

---

## Tóm Tắt Luồng Dữ Liệu End-to-End ⭐⭐⭐

```
execute_workflow("iPhone quá đắt")
    │
    ▼
_build_initial_state()
    → WorkflowState {
        objection_text: "iPhone quá đắt",
        draft_response: "",
        retry_count: 0,
        workflow_status: "initialized"
      }
    │
    ▼
[research_node] — Lần 1
    → Sales Agent viết draft
    → state["draft_response"] = "iPhone 15 giá 35M VND..."
    → state["workflow_status"] = "researching"
    → ExecutionStep appended to execution_log
    │
    ▼
[verification_node] — Lần 1
    → PriceAccuracyChecker: FAIL (35M ≠ 29.99M, deviation=16.7%)
    → PolicyAuthenticityChecker: PASS
    → TopicRelevanceChecker: PASS
    → state["verification_result"] = VerificationResult(is_approved=False)
    → state["workflow_status"] = "correction_needed"
    │
    ▼
route_after_verification()
    → is_approved=False, no CRITICAL, retry_count=0 < max_retries=3
    → return "correction"
    │
    ▼
[correction_node]
    → state["retry_count"] = 1
    → state["correction_feedback"] = """
        🔄 VERIFICATION FAILED - CORRECTION REQUIRED
        💰 PRICE ACCURACY ISSUES:
          - iPhone 15 Pro Max: deviation 16.7%
            💡 Update price to 29,990,000 VND (SKU: IP15PM-256)
        🔄 RETRY INSTRUCTIONS: ...
        ✅ QUALITY CHECKLIST: ...
      """
    │
    ▼
route_after_correction()
    → retry_count=1 < max_retries=3
    → return "retry"
    │
    ▼
[research_node] — Lần 2
    → prompt = correction_feedback + "\n---\nORIGINAL OBJECTION:\niPhone quá đắt"
    → Sales Agent viết lại với giá đúng
    → state["draft_response"] = "iPhone 15 giá 29,990,000 VND..."
    │
    ▼
[verification_node] — Lần 2
    → PriceAccuracyChecker: PASS (29.99M ≈ 29.99M, deviation=0%)
    → PolicyAuthenticityChecker: PASS
    → TopicRelevanceChecker: PASS
    → state["verification_result"] = VerificationResult(is_approved=True)
    → state["workflow_status"] = "approved"
    → state["final_response"] = state["draft_response"]
    │
    ▼
route_after_verification()
    → is_approved=True
    → return "approved"
    │
    ▼
END ✅
    → final_state["workflow_status"] = "approved"
    → final_state["final_response"] = "iPhone 15 giá 29,990,000 VND..."
    → final_state["retry_count"] = 1
    → final_state["execution_log"] = [step1, step2, step3, step4]
```

---

## Key Takeaways — Những điều quan trọng nhất ⭐⭐⭐

**1. Binary > Scoring**
`is_approved = True/False` rõ ràng hơn điểm 0-10. Không có vùng xám, không cần người quyết định.

**2. State là bộ nhớ chung**
Mỗi node đọc state, làm việc, trả về state mới. Không node nào "nhớ" thông tin riêng. Tất cả đều trong state.

**3. Parallel + First-failure-fast**
Chạy 3 checker song song. Nếu gặp CRITICAL, cancel 2 checker còn lại ngay → tiết kiệm 2/3 API calls.

**4. Pure Python Routing**
Routing không dùng AI. Nhanh hơn, rẻ hơn, nhất quán hơn, dễ test hơn.

**5. Structured Feedback**
Correction feedback có cấu trúc rõ ràng (5 sections) giúp Sales Agent biết chính xác cần sửa gì.

**6. Circuit Breaker**
Tự động ngắt mạch khi external service lỗi liên tục. Tránh cascade failure.

**7. Checkpoint/Resume**
Sau mỗi node, tự động save state. Nếu crash, có thể resume từ điểm đã dừng.

**8. Config > Hardcode**
Tất cả thông số đều trong config, không hardcode. Có thể thay đổi mà không cần deploy lại.

---

## Câu Hỏi Tự Kiểm Tra

1. Tại sao `WorkflowState` dùng `TypedDict` thay vì Pydantic?
2. `overall_pass` được tính như thế nào? Tại sao không dùng AI?
3. `early_termination` hoạt động như thế nào? Tiết kiệm được gì?
4. Tại sao routing là pure Python?
5. Circuit Breaker có 3 trạng thái gì? Chuyển đổi như thế nào?
6. Correction feedback có mấy sections? Mỗi section làm gì?
7. Tại sao cần checkpoint sau mỗi node?
8. `verify_draft_sync` khác `verify_draft` như thế nào?
