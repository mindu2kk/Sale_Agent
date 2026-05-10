# SALES RESEARCH AGENT — GIẢI THÍCH TOÀN DIỆN


> Tài liệu này giải thích toàn bộ dự án Sales Research Agent — từ tổng thể kiến trúc đến từng dòng code — theo thứ tự từ **tổng thể → chi tiết → liên hệ ngược lại tổng thể**.

---

## MỤC LỤC

1. [BỨC TRANH TỔNG THỂ — Hệ thống làm gì?](#1-bức-tranh-tổng-thể)
2. [KIẾN TRÚC HỆ THỐNG — Các thành phần và luồng dữ liệu](#2-kiến-trúc-hệ-thống)
3. [TẦNG 1 — RAG Pipeline (Nền tảng tra cứu)](#3-tầng-1--rag-pipeline)
4. [TẦNG 2 — Sales Research Agent (Bộ não ReAct)](#4-tầng-2--sales-research-agent)
5. [TẦNG 3 — Tools (Tay chân của Agent)](#5-tầng-3--tools)
6. [TẦNG 4 — Cache (Bộ nhớ đệm)](#6-tầng-4--cache)
7. [TẦNG 5 — Prompts (Ngôn ngữ điều khiển Agent)](#7-tầng-5--prompts)
8. [TESTING STRATEGY — Cách kiểm tra tính đúng đắn](#8-testing-strategy)
9. [PROPERTY-BASED TESTING — Kiểm tra bằng tính chất](#9-property-based-testing)
10. [LIÊN HỆ TỔNG THỂ — Tất cả kết nối như thế nào](#10-liên-hệ-tổng-thể)
11. [BÀI HỌC RÚT RA — Patterns quan trọng khi xây Agent](#11-bài-học-rút-ra)

---

## 1. BỨC TRANH TỔNG THỂ

### Hệ thống giải quyết vấn đề gì?

Hãy tưởng tượng một nhân viên bán hàng điện tử đang đứng trước khách hàng. Khách nói: *"Máy Dell Inspiron này đắt quá, bên kia bán rẻ hơn."* Nhân viên cần phản hồi ngay, chuyên nghiệp, dựa trên dữ liệu thực tế của công ty.

Hệ thống này **tự động soạn bản nháp phản hồi** cho nhân viên, bằng cách:
1. Tra cứu dữ liệu nội bộ (catalog sản phẩm, chính sách bảo hành)
2. Nếu không đủ, tìm thêm thông tin từ web
3. Tổng hợp thành bản nháp theo văn phong chuyên nghiệp "Dạ/Vâng"

### Tại sao không dùng RAG thuần túy?

**RAG thuần túy** (Retrieval-Augmented Generation) chỉ làm một việc: tìm kiếm → trả lời. Nó **thụ động** — không tự quyết định được khi nào cần tìm thêm, khi nào đã đủ thông tin.

**ReAct Agent** (Reasoning + Acting) thì **chủ động**:
- Tự **suy luận** (Thought): "Tôi cần tìm thông số kỹ thuật của Dell Inspiron"
- Tự **hành động** (Action): Gọi tool `internal_db_search`
- Tự **quan sát** (Observation): Đọc kết quả trả về
- Tự **quyết định**: "Đã đủ thông tin, không cần web search nữa"

```
RAG thuần túy:  Query → Retrieve → Generate → Done
ReAct Agent:    Query → [Thought → Action → Observation] × N → Final Answer
```

### Stack công nghệ

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| LLM Framework | LlamaIndex | Quản lý ReAct loop, tool calling |
| Vector DB | ChromaDB | Lưu embeddings, tìm kiếm ngữ nghĩa |
| Keyword Search | BM25 (rank_bm25) | Tìm kiếm từ khóa chính xác |
| Web Search | Tavily API | Tìm kiếm internet khi cần |
| Testing | Hypothesis (PBT) | Kiểm tra tính chất bằng dữ liệu ngẫu nhiên |
| Embedding | BAAI/bge-m3 | Chuyển text thành vector |

---

## 2. KIẾN TRÚC HỆ THỐNG

### Sơ đồ luồng dữ liệu tổng thể

```
[Nhân viên bán hàng]
        |
        | "Khách chê Dell Inspiron đắt"
        v
[SalesResearchAgent.run()]
        |
        v
[ReAct Loop - max 2 vòng]
   |
   |-- Thought: "Cần tìm thông tin Dell Inspiron"
   |-- Action: internal_db_search("Dell Inspiron thông số giá")
   |       |
   |       v
   |   [RAGPipeline.query()]
   |       |
   |       |-- [RelevanceChecker] → "CAN_ANSWER"
   |       |-- [HybridRetriever.retrieve()]
   |               |
   |               |-- BM25 Search (song song)
   |               |-- Vector Search (song song)
   |               |-- RRF Fusion
   |               |-- SKU Boost
   |               v
   |           [Danh sách nodes có liên quan]
   |
   |-- Observation: JSON với thông số Dell Inspiron
   |-- Thought: "Đã đủ thông tin, không cần web search"
   |-- Final Answer: "Dạ, em hiểu anh/chị đang so sánh về giá..."
        |
        v
[AgentResult]
   - objection_text: "Khách chê Dell Inspiron đắt"
   - draft_response: "Dạ, em hiểu..."
   - tools_used: ["internal_db_search"]
        |
        v
[Verification Agent - Tuần 6] (chưa implement trong spec này)
```

### Cấu trúc thư mục và vai trò từng file

```
agent/
├── sales_research_agent.py  ← TRUNG TÂM: AgentResult + SalesResearchAgent
├── tools.py                 ← HAI TAY: build_internal_db_tool, build_tavily_tool
├── prompts.py               ← NGÔN NGỮ: AGENT_SYSTEM_PROMPT + build_correction_context
└── cache.py                 ← BỘ NHỚ ĐỆM: QueryCache (LRU + TTL)

retriever/
├── hybrid_retriever.py      ← TÌM KIẾM HYBRID: BM25 + Vector + RRF
└── relevance_checker.py     ← BỘ LỌC: Phân loại query trước khi tìm kiếm

rag_pipeline.py              ← ĐIỀU PHỐI: Kết nối checker + retriever

tests/
├── test_unit.py             ← Unit tests cho retriever + agent
├── test_pbt.py              ← Property-based tests (Hypothesis)
└── test_integration.py      ← Integration tests

agent/
├── test_sales_research_agent_correction.py  ← Test correction loop
└── test_agent_result_workflow_integration.py ← Test AgentResult fields
```

---

## 3. TẦNG 1 — RAG Pipeline (Nền tảng tra cứu)

Đây là **nền tảng** mà Agent dựa vào để tra cứu dữ liệu nội bộ. Gồm 3 file.

---

### 3.1 `retriever/relevance_checker.py` — Bộ lọc thông minh

**Mục đích:** Trước khi tìm kiếm tốn kém, hỏi LLM xem câu hỏi có liên quan không. Tiết kiệm chi phí API và thời gian.

```python
# File: retriever/relevance_checker.py

RelevanceLabel = Literal["CAN_ANSWER", "PARTIAL", "NO_MATCH"]
# 3 nhãn có thể trả về:
# CAN_ANSWER: câu hỏi về sản phẩm/giá/chính sách → tìm kiếm đầy đủ
# PARTIAL:    câu hỏi mơ hồ, liên quan một phần → vẫn tìm kiếm
# NO_MATCH:   hoàn toàn không liên quan → bỏ qua, trả về thông báo mặc định
```

**Phân tích từng phần quan trọng:**

```python
_SYSTEM_PROMPT_TEMPLATE = """Bạn là bộ phân loại câu hỏi...
Câu hỏi: {query}
Nhãn:"""
```
> **Lưu ý:** Prompt chỉ yêu cầu trả về 1 nhãn, không giải thích. Đây là kỹ thuật **constrained output** — ép LLM trả về đúng format mình muốn.

```python
def check(self, query: str) -> RelevanceLabel:
    if query.strip() == "":
        return "NO_MATCH"          # ← Xử lý edge case: query rỗng
    
    prompt = self._prompt_template.format(query=query)
    response = self.llm.complete(prompt)
    raw = response.text.strip().rstrip(".!?").upper()  # ← Normalize output
    
    valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
    if raw in valid_labels:
        return raw
    return "PARTIAL"               # ← Fallback an toàn: nếu LLM trả về lạ → PARTIAL
```

> **Điểm quan trọng:** `.rstrip(".!?").upper()` — LLM đôi khi trả về "can_answer." hoặc "CAN_ANSWER!" → normalize hết về dạng chuẩn. Fallback về "PARTIAL" thay vì crash.

---

### 3.2 `retriever/hybrid_retriever.py` — Tìm kiếm lai BM25 + Vector

**Mục đích:** Kết hợp 2 phương pháp tìm kiếm để bù đắp điểm yếu của nhau:
- **BM25**: Giỏi tìm từ khóa chính xác (SKU, tên sản phẩm cụ thể)
- **Vector Search**: Giỏi tìm ngữ nghĩa (câu hỏi diễn đạt khác nhau nhưng cùng ý)

**Khởi tạo — `__init__`:**

```python
def __init__(
    self,
    docstore_path: str = "./chroma_db/docstore.json",  # File JSON chứa tất cả nodes
    chroma_path: str = "./chroma_db",                   # Thư mục ChromaDB
    collection_name: str = "sales_copilot_vdb",         # Tên collection trong ChromaDB
    embed_model_name: str = "BAAI/bge-m3",              # Model embedding đa ngôn ngữ
    bm25_top_k: int = 20,    # Lấy top 20 từ BM25
    vector_top_k: int = 20,  # Lấy top 20 từ Vector
    rrf_k: int = 60,         # Hằng số RRF (chuẩn trong literature)
) -> None:
```

**`_init_bm25` — Xây dựng BM25 index:**

```python
def _init_bm25(self) -> None:
    # 1. Đọc docstore.json — file chứa tất cả text nodes đã được ingested
    with open(self.docstore_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docstore_data = data.get("docstore/data", {})
    
    # 2. Tạo TextNode objects từ JSON
    nodes = []
    for node_id, node_entry in docstore_data.items():
        node_data = node_entry.get("__data__", {})
        nodes.append(TextNode(**node_data))
    
    self._nodes = nodes
    
    # 3. Tokenize: tách từng node thành list of words
    self._bm25_corpus = [node.text.split() for node in self._nodes]
    
    # 4. Build BM25Okapi index
    if self._bm25_corpus:
        self._bm25 = BM25Okapi(self._bm25_corpus)
    else:
        self._bm25 = BM25Okapi([[""]])  # ← Dummy để tránh crash khi corpus rỗng
```

> **Lưu ý:** BM25Okapi yêu cầu ít nhất 1 document. Nếu corpus rỗng, dùng `[[""]]` làm dummy — đây là defensive programming.

**`retrieve` — Hàm chính, chạy song song:**

```python
def retrieve(self, query: str, top_k: int = 10) -> list[NodeWithScore]:
    # BƯỚC 1: Chạy BM25 và Vector Search SONG SONG (ThreadPoolExecutor)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25 = executor.submit(self._bm25_search, query)
        future_vector = executor.submit(self._vector_search, query)
        bm25_results = future_bm25.result()
        vector_results = future_vector.result()
    
    # BƯỚC 2: RRF Fusion — gộp kết quả từ 2 nguồn
    fusion_scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}
    
    for result_list in (bm25_results, vector_results):
        for rank, nws in enumerate(result_list, start=1):
            node_id = nws.node.node_id
            # Công thức RRF: score += 1 / (k + rank)
            # rank=1 → 1/61 ≈ 0.016, rank=2 → 1/62 ≈ 0.016, ...
            # Node xuất hiện ở cả 2 nguồn → cộng dồn → điểm cao hơn
            fusion_scores[node_id] = fusion_scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
            if node_id not in node_map:
                node_map[node_id] = nws
    
    # BƯỚC 3: Sort theo fusion score, tie-break bằng source_type
    source_priority = {"product_catalog": 1, "policy_pdf": 0}
    # product_catalog ưu tiên hơn policy_pdf khi điểm bằng nhau
    
    sorted_ids = sorted(fusion_scores.keys(), key=sort_key, reverse=True)
    results = [NodeWithScore(node=node_map[nid].node, score=fusion_scores[nid])
               for nid in sorted_ids]
    
    # BƯỚC 4: Cắt top_k
    results = results[:top_k]
    
    # BƯỚC 5: SKU Boost — nếu query chứa product_code, đẩy node đó lên đầu
    query_tokens = query.split()
    for i, nws in enumerate(results):
        product_code = nws.node.metadata.get("product_code", "")
        if product_code and product_code in query_tokens:
            boosted = results.pop(i)
            results.insert(0, boosted)  # ← Đẩy lên vị trí 0
            break
    
    return results
```

> **Tại sao RRF?** Vì BM25 và Vector Search dùng thang điểm khác nhau (BM25 dùng TF-IDF score, Vector dùng cosine similarity). Không thể cộng trực tiếp. RRF chỉ dùng **thứ hạng** (rank), không dùng điểm số tuyệt đối → chuẩn hóa được.

> **Tại sao ThreadPoolExecutor?** BM25 là CPU-bound, Vector Search là I/O-bound (gọi ChromaDB). Chạy song song giảm latency đáng kể.

---

### 3.3 `rag_pipeline.py` — Điều phối tổng thể

**Mục đích:** Kết nối RelevanceChecker và HybridRetriever thành một pipeline hoàn chỉnh.

```python
class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, checker: RelevanceChecker) -> None:
        self.retriever = retriever
        self.checker = checker
    
    def query(self, user_query: str) -> Union[str, list[NodeWithScore]]:
        # BƯỚC 1: Phân loại query
        label = self.checker.check(user_query)
        logger.info(f"[RelevanceChecker] label={label} query={user_query!r}")
        
        # BƯỚC 2: Route dựa trên label
        if label == "NO_MATCH":
            return _DEFAULT_NO_MATCH_RESPONSE  # ← Trả về string, không tìm kiếm
        
        # CAN_ANSWER hoặc PARTIAL → tìm kiếm
        return self.retriever.retrieve(user_query)  # ← Trả về list[NodeWithScore]
```

> **Điểm quan trọng:** Return type là `Union[str, list[NodeWithScore]]` — 2 kiểu khác nhau! Đây là lý do `internal_db_search` trong tools.py phải kiểm tra `isinstance(result, str)` để xử lý NO_MATCH.

---

## 4. TẦNG 2 — Sales Research Agent (Bộ não ReAct)

### 4.1 `agent/sales_research_agent.py` — File trung tâm

**Mục đích:** Định nghĩa `AgentResult` (output contract) và `SalesResearchAgent` (bộ não chính).

---

#### AgentResult — Output Contract

```python
@dataclass
class AgentResult:
    # --- CORE FIELDS (bắt buộc, luôn có) ---
    objection_text: str          # Input gốc từ nhân viên (không bao giờ thay đổi)
    draft_response: str          # Bản nháp do Agent sinh ra
    tools_used: list[str] = field(default_factory=list)  # Danh sách tools đã gọi

    # --- WORKFLOW INTEGRATION FIELDS (tùy chọn, cho Verification Agent) ---
    verification_result: Optional["VerificationResult"] = field(default=None)
    workflow_status: Optional[WorkflowStatus] = field(default=None)
    retry_count: int = field(default=0)
    correction_feedback: Optional[str] = field(default=None)
```

> **Tại sao dùng `@dataclass`?**
> - Tự động sinh `__init__`, `__repr__`, `__eq__`
> - Hỗ trợ `dataclasses.asdict()` để serialize thành dict cho LangGraph StateGraph
> - `field(default_factory=list)` tạo list mới cho mỗi instance (tránh shared mutable default — bug kinh điển Python)

> **Tại sao `TYPE_CHECKING` cho VerificationResult?**
> ```python
> if TYPE_CHECKING:
>     from verification.models.verification import VerificationResult
> ```
> Import chỉ chạy khi type checker (mypy/pyright) phân tích, không chạy lúc runtime. Tránh circular import giữa `agent` và `verification` modules.

> **WorkflowStatus là gì?**
> ```python
> WorkflowStatus = Literal["initialized", "researching", "verifying",
>                           "correcting", "approved", "escalated", "failed"]
> ```
> Đây là các trạng thái trong LangGraph StateGraph (Tuần 6). Agent ở Tuần 5 chưa dùng nhưng đã chuẩn bị sẵn để tích hợp.

---

#### SalesResearchAgent — Khởi tạo

```python
class SalesResearchAgent:
    def __init__(self, llm, rag_pipeline, tavily_api_key: str | None = None) -> None:
        # VALIDATION: Fail fast — báo lỗi ngay khi thiếu dependency
        if llm is None:
            raise ValueError("llm cannot be None")
        if rag_pipeline is None:
            raise ValueError("rag_pipeline cannot be None")
        
        # BUILD TOOLS LIST
        tools = [build_internal_db_tool(rag_pipeline)]  # ← Luôn có Internal DB
        
        tavily_tool = build_tavily_tool(tavily_api_key)
        if tavily_tool is not None:
            tools.append(tavily_tool)  # ← Chỉ thêm nếu có API key
        
        # KHỞI TẠO ReActAgent
        self._agent = ReActAgent.from_tools(
            tools,
            llm=llm,
            verbose=True,
            max_iterations=2,          # ← LOOP CONTROL: tối đa 2 lần gọi tool
            context=AGENT_SYSTEM_PROMPT,  # ← System prompt định hình hành vi
        )
```

> **`max_iterations=2` — Tại sao quan trọng?**
> - Kiểm soát chi phí API: mỗi tool call = 1 API request
> - Đảm bảo response time < 10 giây
> - Tránh infinite loop khi LLM "bị lạc" trong reasoning
> - Với 2 tools (internal_db + tavily), max 2 iterations là đủ

---

#### SalesResearchAgent — Phương thức `run()`

```python
def run(
    self,
    objection: str,
    correction_feedback: Optional[str] = None,    # ← Từ Verification Agent
    verification_issues: Optional[list] = None,   # ← Chi tiết lỗi cụ thể
) -> AgentResult:
    try:
        # BƯỚC 1: Xây dựng query
        if correction_feedback:
            # Đây là lần RETRY — prepend correction context
            correction_ctx = build_correction_context(
                correction_feedback=correction_feedback,
                verification_issues=verification_issues,
            )
            query = f"{correction_ctx}\n\n---\n\nCÂU HỎI GỐC CỦA KHÁCH HÀNG:\n{objection}"
            logger.info("Running agent with correction feedback for objection: %s", objection)
        else:
            # Lần đầu tiên — query thẳng
            query = objection
            logger.info("Running agent for objection: %s", objection)
        
        # BƯỚC 2: Gọi ReAct Agent
        response = self._agent.chat(query)
        
        # BƯỚC 3: Extract tool usage từ response.sources
        tools_used = [s.tool_name for s in response.sources]
        
        # BƯỚC 4: Log từng tool call
        for source in response.sources:
            logger.info("Tool called: %s | Input: %s", source.tool_name, source.raw_input)
            logger.info("Observation: %s", source.raw_output)
        
        # BƯỚC 5: Trả về AgentResult
        return AgentResult(
            objection_text=objection,    # ← LUÔN là input gốc, không phải query đã modify
            draft_response=str(response),
            tools_used=tools_used,
        )
    
    except Exception as exc:
        # BƯỚC 6: Error handling — KHÔNG BAO GIỜ crash
        logger.error("Agent error: %s", exc, exc_info=True)
        return AgentResult(
            objection_text=objection,
            draft_response="Dạ, hiện tại hệ thống đang gặp sự cố kỹ thuật. Xin phép ghi nhận để báo cáo quản lý ạ.",
            tools_used=[],
        )
```

> **3 điểm cực kỳ quan trọng:**
>
> 1. **`objection_text=objection` (không phải `query`)**: Dù query có thể được prepend correction context, `objection_text` luôn lưu input gốc. Đây là **round-trip property** — Property 9 trong PBT.
>
> 2. **`response.sources`**: LlamaIndex ReActAgent lưu lịch sử tool calls trong `response.sources`. Mỗi source có `tool_name`, `raw_input`, `raw_output`. Đây là cách duy nhất để biết agent đã gọi tool nào.
>
> 3. **`except Exception`**: Bắt TẤT CẢ exceptions, kể cả LLM timeout, network error, v.v. Trả về fallback message bắt đầu bằng "Dạ," — đảm bảo Property 8 (Error Resilience).

---

## 5. TẦNG 3 — Tools (Tay chân của Agent)

### 5.1 `agent/tools.py` — Hai công cụ của Agent

**Mục đích:** Wrap các external services thành `FunctionTool` mà LlamaIndex ReActAgent có thể gọi.

---

#### `build_internal_db_tool` — Factory Pattern

```python
def build_internal_db_tool(rag_pipeline, cache=None):
    """Factory tạo FunctionTool với closure over rag_pipeline."""
    
    def internal_db_search(query: str) -> str:
        """Docstring này QUAN TRỌNG — LLM đọc để quyết định có gọi tool không."""
        try:
            # BƯỚC 1: Kiểm tra cache trước
            if cache is not None:
                cached = cache.get(query)
                if cached is not None:
                    logging.getLogger(__name__).debug("Cache hit for query: %s", query)
                    return cached  # ← Trả về ngay, không gọi RAGPipeline
            
            # BƯỚC 2: Gọi RAGPipeline
            result = rag_pipeline.query(query)
            
            # BƯỚC 3: Xử lý kết quả
            if isinstance(result, str):
                # RAGPipeline trả về string → NO_MATCH
                result_json = json.dumps({"status": "NO_MATCH", "message": result})
            else:
                # RAGPipeline trả về list[NodeWithScore] → format thành JSON
                formatted_nodes = []
                for nws in result:
                    formatted_nodes.append({
                        "source": nws.node.metadata.get("source_type", "Unknown"),
                        "product_code": nws.node.metadata.get("product_code", "N/A"),
                        "content": nws.node.text[:500] + "...",  # ← TRUNCATE 500 ký tự
                    })
                result_json = json.dumps(formatted_nodes, ensure_ascii=False)
            
            # BƯỚC 4: Lưu vào cache
            if cache is not None:
                cache.set(query, result_json)
            
            return result_json
        
        except Exception as e:
            # BƯỚC 5: Error handling — trả về JSON error, không crash
            return json.dumps({"status": "ERROR", "message": str(e)})
    
    # Đăng ký thành FunctionTool
    return FunctionTool.from_defaults(
        fn=internal_db_search,
        name="internal_db_search",
        description=(
            "Tra cứu thông tin sản phẩm, giá, thông số kỹ thuật và chính sách "
            "bảo hành/đổi trả từ cơ sở dữ liệu nội bộ của công ty. "
            "Luôn gọi tool này TRƯỚC khi dùng web search."  # ← Micro-prompt trong description
        ),
    )
```

> **Tại sao dùng Factory Pattern (`build_internal_db_tool`)?**
> - `rag_pipeline` và `cache` được **capture trong closure** của `internal_db_search`
> - Mỗi lần gọi factory tạo ra một tool mới với pipeline riêng
> - Dễ test: có thể inject mock pipeline và mock cache

> **Tại sao truncate 500 ký tự?**
> - LLM có context window giới hạn (ví dụ GPT-4: 128K tokens)
> - Mỗi node có thể dài hàng nghìn ký tự
> - Với max 2 tool calls, nếu mỗi call trả về 10 nodes × 500 ký tự = 5000 ký tự → vẫn trong giới hạn
> - Đây là **Memory Safety** — tránh Context Bloat

> **Tại sao `ensure_ascii=False`?**
> - Dữ liệu tiếng Việt có ký tự Unicode (ạ, ề, ổ, ...)
> - `ensure_ascii=True` (mặc định) sẽ encode thành `\u1ea1` → LLM khó đọc hơn
> - `ensure_ascii=False` giữ nguyên ký tự Unicode → LLM đọc được tiếng Việt

---

#### `build_tavily_tool` — Conditional Tool

```python
def build_tavily_tool(tavily_api_key: str | None = None):
    """Trả về None nếu không có API key — tool này là tùy chọn."""
    
    # Kiểm tra API key từ tham số hoặc environment variable
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None  # ← Không có key → không tạo tool
    
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)  # ← Khởi tạo client trong closure
    
    def tavily_web_search(query: str) -> str:
        """CHỈ dùng khi Internal_DB_Tool hoàn toàn không có thông tin."""
        try:
            response = client.search(query, max_results=3)  # ← Giới hạn 3 kết quả
            results = response.get("results", [])
            return json.dumps(
                [{"title": r.get("title"), "content": r.get("content", "")[:500]}
                 for r in results],
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": str(e)})
    
    return FunctionTool.from_defaults(
        fn=tavily_web_search,
        name="tavily_web_search",
        description=(
            "...CHỈ dùng khi Internal_DB_Tool hoàn toàn không có thông tin..."
        ),
    )
```

> **Tại sao `from tavily import TavilyClient` nằm BÊN TRONG hàm?**
> - Nếu `tavily` không được cài, import ở top-level sẽ crash toàn bộ module
> - Import bên trong hàm chỉ chạy khi hàm được gọi VÀ có API key
> - Đây là **lazy import** — tránh dependency bắt buộc

> **Tại sao description của tool quan trọng?**
> - LlamaIndex ReActAgent đọc description để quyết định gọi tool nào
> - "CHỈ dùng khi Internal_DB_Tool hoàn toàn không có thông tin" → đây là **Micro-prompt trong tool description**
> - Kết hợp với System Prompt → double enforcement của Smart Routing

---

## 6. TẦNG 4 — Cache (Bộ nhớ đệm)

### 6.1 `agent/cache.py` — LRU Cache với TTL

**Mục đích:** Tránh gọi RAGPipeline nhiều lần với cùng query. Tiết kiệm chi phí và tăng tốc độ.

```python
class QueryCache:
    """In-memory LRU cache với TTL expiry và max_size eviction."""
    
    def __init__(self, ttl: float = 300.0, max_size: int = 100) -> None:
        self._ttl = ttl          # Time-To-Live: 300 giây = 5 phút
        self._max_size = max_size  # Tối đa 100 entries
        # OrderedDict: dict có thứ tự → O(1) evict entry cũ nhất
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        #                                  ↑ value  ↑ timestamp
```

**`get` — Đọc từ cache:**

```python
def get(self, key: str) -> Optional[str]:
    entry = self._store.get(key)
    if entry is None:
        return None  # Cache miss
    
    value, ts = entry
    if time.time() - ts >= self._ttl:
        del self._store[key]  # ← Xóa entry đã hết hạn
        return None           # Cache miss (expired)
    
    logger.debug("Cache hit for query: %s", key)
    return value  # Cache hit
```

**`set` — Ghi vào cache:**

```python
def set(self, key: str, value: str) -> None:
    if key in self._store:
        del self._store[key]  # ← Xóa để re-insert ở cuối (most-recently-used)
    elif len(self._store) >= self._max_size:
        self._store.popitem(last=False)  # ← Evict entry ĐẦU TIÊN (oldest/LRU)
    
    self._store[key] = (value, time.time())  # ← Thêm vào CUỐI (most-recently-used)
```

> **Tại sao dùng `OrderedDict` thay vì `dict` thường?**
> - Python 3.7+ `dict` có thứ tự insertion, nhưng `OrderedDict` có `popitem(last=False)` để lấy item đầu tiên (oldest) trong O(1)
> - Đây là cách implement **LRU (Least Recently Used)** đơn giản nhất
> - Khi `set` một key đã tồn tại: xóa rồi re-insert → key đó được đẩy về cuối (most-recently-used)
> - Khi cần evict: `popitem(last=False)` lấy item đầu tiên (least-recently-used)

> **TTL vs LRU — Hai cơ chế độc lập:**
> - **TTL**: Entry hết hạn sau 300 giây dù có được dùng hay không → đảm bảo data freshness
> - **LRU**: Khi đầy, xóa entry ít được dùng nhất → đảm bảo memory không tràn
> - Cả hai cùng hoạt động: entry có thể bị xóa vì TTL hoặc vì LRU eviction

---

## 7. TẦNG 5 — Prompts (Ngôn ngữ điều khiển Agent)

### 7.1 `agent/prompts.py` — System Prompt và Correction Context

**Mục đích:** Định hình toàn bộ hành vi của Agent thông qua ngôn ngữ tự nhiên.

---

#### AGENT_SYSTEM_PROMPT — Phân tích từng section

```
Bạn là chuyên viên tư vấn bán hàng điện tử chuyên nghiệp của công ty FPTS.
```
> **Persona**: Đặt vai trò rõ ràng → LLM "nhập vai" tốt hơn, văn phong phù hợp hơn.

```
## QUY TẮC SỐNG CÒN (BẮT BUỘC TUÂN THỦ)
1. ƯU TIÊN NỘI BỘ: LUÔN gọi `internal_db_search` TRƯỚC TIÊN...
```
> **Micro-Prompt / Smart Routing**: Dùng từ ngữ mạnh ("LUÔN", "TRƯỚC TIÊN", "BẮT BUỘC") để ép LLM tuân thủ. Đây là kỹ thuật **instruction following** — LLM được train để tuân theo instructions rõ ràng.

```
2. CÔNG TY LÀ CHÂN LÝ: Dữ liệu từ `internal_db_search` là tuyệt đối...
   Tuyệt đối không tự ý giảm giá theo đối thủ.
```
> **Conflict Resolution Rule**: Khi có xung đột giữa dữ liệu nội bộ và web → luôn dùng nội bộ. Đây là business rule quan trọng nhất — bảo vệ công ty khỏi việc agent tự ý giảm giá.

```
5. LỐI THOÁT HIỂM (FALLBACK): Nếu cả 2 công cụ đều không tìm thấy...
   TUYỆT ĐỐI KHÔNG BỊA ĐẶT.
```
> **Hallucination Prevention**: LLM có xu hướng "bịa" thông tin khi không có dữ liệu. Rule này ép agent thừa nhận không biết thay vì bịa đặt.

```
6. KẾT THÚC SỚM: Nếu `internal_db_search` đã trả về thông tin đầy đủ...
   sinh Final Answer ngay — KHÔNG gọi thêm tool.
```
> **Early Termination**: Tránh gọi Tavily khi không cần thiết → tiết kiệm chi phí và thời gian.

```
## VĂN PHONG & CẤU TRÚC
- Luôn bắt đầu câu trả lời bằng "Dạ," hoặc "Vâng,"
- Thể hiện sự thấu hiểu tâm lý khách hàng (Empathy)
- Bắt buộc trích dẫn ít nhất 1 thông số kỹ thuật...
- Giới hạn độ dài trong khoảng 150-300 từ.
```
> **Output Format Control**: Định nghĩa chính xác format output → dễ test (Property 3: draft phải bắt đầu "Dạ," hoặc "Vâng,").

---

#### `build_correction_context` — Correction Loop

```python
CORRECTION_CONTEXT_TEMPLATE = """
⚠️ LƯU Ý QUAN TRỌNG: Đây là lần thử lại sau khi bản nháp trước bị từ chối...

{correction_feedback}

Hãy đảm bảo bản nháp mới khắc phục TẤT CẢ các vấn đề...
"""

def build_correction_context(
    correction_feedback: str,
    verification_issues: list | None = None,
) -> str:
    context = CORRECTION_CONTEXT_TEMPLATE.format(correction_feedback=correction_feedback)
    
    if verification_issues:
        issue_lines = ["📋 CHI TIẾT CÁC VẤN ĐỀ CẦN SỬA:"]
        for issue in verification_issues:
            issue_type = type(issue).__name__
            
            if issue_type == "PriceIssue":
                # Ví dụ: "• [GIÁ] iPhone 15: đề cập '35M', thực tế '30M' (sai lệch 16.7%)"
                line = f"  • [GIÁ] {issue.product_name}: đề cập '{issue.mentioned_price}'..."
            
            elif issue_type == "PolicyIssue":
                # Ví dụ: "• [CHÍNH SÁCH BỊA ĐẶT] warranty: 'Bảo hành 3 năm' → đúng là: 'Bảo hành 1 năm'"
                fabricated_tag = " [BỊA ĐẶT]" if issue.is_fabricated else ""
                line = f"  • [CHÍNH SÁCH{fabricated_tag}] {issue.policy_type}..."
            
            elif issue_type == "RelevanceIssue":
                # Ví dụ: "• [ĐỘ PHÙ HỢP] Coverage 45%: ... | Thiếu: camera, battery"
                line = f"  • [ĐỘ PHÙ HỢP] Coverage {issue.response_coverage:.0%}..."
    
    return context
```

> **Tại sao cần Correction Context?**
> - Verification Agent (Tuần 6) có thể từ chối bản nháp vì sai giá, sai chính sách, hoặc không đủ thông tin
> - Khi retry, Agent cần biết **chính xác** vấn đề là gì để sửa
> - Thay vì chỉ nói "sai rồi, làm lại", correction context nói "giá iPhone 15 bạn đề cập là 35M nhưng thực tế là 30M, sai lệch 16.7%"
> - Đây là **structured feedback** — cụ thể hơn nhiều so với feedback chung chung

---

## 8. TESTING STRATEGY — Cách kiểm tra tính đúng đắn

### 8.1 Tổng quan chiến lược test

Dự án dùng **3 tầng test** theo kim tự tháp:

```
        /\
       /  \
      / E2E \        ← Ít nhất, chậm nhất, test toàn bộ hệ thống
     /--------\
    / Integration\   ← Test kết hợp nhiều components
   /--------------\
  /   Unit Tests   \  ← Nhiều nhất, nhanh nhất, test từng component
 /------------------\
/ Property-Based Tests\ ← Đặc biệt: test với dữ liệu ngẫu nhiên
```

---

### 8.2 Unit Tests — `tests/test_unit.py`

**Nguyên tắc:** Mock tất cả external dependencies, chỉ test logic của component đang xét.

**Ví dụ quan trọng — test tool exception:**

```python
def test_tool_exception_returns_safe_json():
    """Mock rag_pipeline.query raise RuntimeError → internal_db_search phải trả về
    JSON với status==ERROR, không propagate exception."""
    
    mock_pipeline = MagicMock()
    mock_pipeline.query.side_effect = RuntimeError("db failure")  # ← Inject exception
    
    tool = build_internal_db_tool(mock_pipeline)
    result_json = tool.fn("any query")  # ← Gọi trực tiếp hàm bên trong tool
    
    result = json.loads(result_json)
    assert result["status"] == "ERROR"
    assert "db failure" in result["message"]
    # Không có exception được raise → test pass
```

> **`tool.fn`**: FunctionTool của LlamaIndex expose hàm gốc qua `.fn`. Gọi trực tiếp để test logic mà không cần khởi tạo toàn bộ Agent.

**Ví dụ quan trọng — test cache:**

```python
def test_cache_ttl_expiry():
    cache = QueryCache(ttl=0.01)  # ← TTL 10ms để test nhanh
    cache.set("q1", "value")
    time.sleep(0.02)              # ← Chờ hết TTL
    assert cache.get("q1") is None  # ← Phải là None (expired)

def test_cache_max_size_eviction():
    cache = QueryCache(max_size=3)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")
    cache.set("k4", "v4")  # ← Thêm entry thứ 4 → evict k1 (oldest)
    assert cache.get("k1") is None  # ← k1 đã bị evict
    assert cache.get("k4") == "v4"  # ← k4 vẫn còn
```

---

### 8.3 Test Correction Loop — `agent/test_sales_research_agent_correction.py`

**Mục đích:** Verify rằng correction feedback được inject đúng vào query.

```python
def test_correction_feedback_prepended_to_query(self):
    agent = _make_agent()
    agent._agent.chat.return_value = _mock_chat_response("Dạ, bản nháp sửa.")
    
    result = agent.run(
        "iPhone quá đắt",
        correction_feedback="Fix the price of iPhone 15.",
    )
    
    # Kiểm tra query được truyền vào agent.chat
    call_args = agent._agent.chat.call_args[0][0]
    assert "Fix the price of iPhone 15." in call_args  # ← Correction context có trong query
    assert "iPhone quá đắt" in call_args               # ← Objection gốc vẫn còn
    
    # Nhưng objection_text trong result vẫn là input gốc
    assert result.objection_text == "iPhone quá đắt"
```

> **`agent._agent.chat.call_args[0][0]`**: Đây là cách kiểm tra argument được truyền vào mock. `call_args[0]` là positional args, `[0]` là argument đầu tiên (query string).

---

### 8.4 Test AgentResult Integration — `agent/test_agent_result_workflow_integration.py`

**Mục đích:** Verify backward compatibility và workflow integration fields.

```python
def test_tools_used_default_is_independent_per_instance(self):
    """Mỗi instance có list riêng — không share mutable default."""
    r1 = AgentResult("q1", "a1")
    r2 = AgentResult("q2", "a2")
    r1.tools_used.append("tool_x")
    assert r2.tools_used == []  # ← r2 không bị ảnh hưởng
```

> **Đây là test cho một bug kinh điển Python:** Nếu dùng `tools_used: list = []` thay vì `field(default_factory=list)`, tất cả instances sẽ SHARE cùng một list. Khi r1 append, r2 cũng thấy. `default_factory=list` tạo list MỚI cho mỗi instance.

---

## 9. PROPERTY-BASED TESTING — Kiểm tra bằng tính chất

### 9.1 PBT là gì và tại sao dùng?

**Unit test thông thường:** Bạn nghĩ ra 5-10 test cases cụ thể và kiểm tra.

**Property-Based Testing:** Bạn định nghĩa một **tính chất** (property) phải đúng với MỌI input, rồi Hypothesis tự sinh hàng trăm input ngẫu nhiên để tìm counterexample.

```python
# Unit test thông thường — chỉ test 1 case
def test_objection_preserved():
    result = agent.run("iPhone quá đắt")
    assert result.objection_text == "iPhone quá đắt"

# Property-Based Test — test với MỌI string
@given(objection=st.text(min_size=0, max_size=500))
@settings(max_examples=30)
def test_pbt_objection_preservation(mock_react_cls, objection: str):
    """For ANY string x, AgentResult.objection_text == x."""
    result = agent.run(objection)
    assert result.objection_text == objection
    # Hypothesis sẽ thử: "", "a", "abc123", "Khách chê đắt", "🎉", "\n\t", ...
```

---

### 9.2 Phân tích 10 Properties của dự án

**Property 1 — Smart Routing (Internal DB Priority):**
```python
@given(objection=st.text(min_size=1, max_size=200))
def test_pbt_internal_db_priority(mock_react_cls, objection):
    # Mock agent luôn gọi cả 2 tools theo đúng thứ tự
    mock_agent_instance.chat.return_value = make_mock_response(
        "Dạ, đây là thông tin.",
        ["internal_db_search", "tavily_web_search"],  # ← Thứ tự đúng
    )
    result = agent.run(objection)
    
    if "internal_db_search" in result.tools_used and "tavily_web_search" in result.tools_used:
        db_idx = result.tools_used.index("internal_db_search")
        tavily_idx = result.tools_used.index("tavily_web_search")
        assert db_idx < tavily_idx  # ← internal_db phải trước tavily
```
> **Lưu ý:** Test này mock response để luôn trả về đúng thứ tự. Mục đích là verify rằng `AgentResult.tools_used` được populate đúng từ `response.sources`, không phải verify LLM behavior.

**Property 4 — AgentResult Completeness (Never Crash):**
```python
@given(objection=st.text(min_size=0, max_size=500))
def test_pbt_agent_result_completeness(mock_react_cls, objection):
    """For ANY string input (kể cả rỗng, Unicode, ký tự đặc biệt),
    run() KHÔNG raise exception và trả về AgentResult đủ fields."""
    result = agent.run(objection)
    
    assert isinstance(result, AgentResult)
    assert result.objection_text is not None
    assert result.draft_response is not None
    assert result.tools_used is not None
```
> **Đây là property quan trọng nhất** — đảm bảo hệ thống không bao giờ crash với bất kỳ input nào.

**Property 6 — Memory Safety (Truncation):**
```python
@given(node_text=st.text(min_size=1, max_size=2000))
def test_pbt_memory_safety_truncation(node_text: str) -> None:
    """For ANY node text (1-2000 chars), content trong JSON output có len <= 503."""
    mock_node.node.text = node_text
    mock_pipeline.query.return_value = [mock_node]
    
    tool = build_internal_db_tool(mock_pipeline)
    result_json = tool.fn("test query")
    
    nodes = json.loads(result_json)
    assert len(nodes[0]["content"]) <= 503  # 500 ký tự + "..."
```
> **Tại sao 503?** `text[:500] + "..."` → tối đa 500 + 3 = 503 ký tự.

**Property 7 — Cache Idempotence:**
```python
@given(query=st.text(min_size=1, max_size=100))
def test_pbt_cache_idempotence(query: str) -> None:
    """Gọi 2 lần cùng query → kết quả giống nhau, RAGPipeline chỉ gọi 1 lần."""
    cache = QueryCache()
    tool = build_internal_db_tool(mock_pipeline, cache=cache)
    
    result1 = tool.fn(query)
    result2 = tool.fn(query)
    
    assert result1 == result2                    # ← Idempotent
    mock_pipeline.query.assert_called_once()     # ← Chỉ gọi RAGPipeline 1 lần
```

**Property 8 — Error Resilience:**
```python
@given(objection=st.text(min_size=1, max_size=200))
def test_pbt_error_resilience(mock_react_cls, objection: str) -> None:
    """For ANY exception từ LLM, run() trả về AgentResult với draft bắt đầu 'Dạ,'."""
    mock_agent_instance.chat.side_effect = RuntimeError("LLM failure")  # ← Inject exception
    
    result = agent.run(objection)
    
    assert isinstance(result, AgentResult)
    assert result.draft_response.startswith("Dạ,")  # ← Fallback message
```

---

### 9.3 Cách Hypothesis hoạt động

```python
@given(st.text(min_size=0, max_size=500))
@settings(max_examples=30)
def test_something(text: str):
    ...
```

1. Hypothesis sinh 30 string ngẫu nhiên (min_size=0, max_size=500)
2. Chạy test với từng string
3. Nếu test fail với string X → Hypothesis **shrink** X xuống string nhỏ nhất vẫn fail
4. Báo cáo counterexample nhỏ nhất → dễ debug hơn

> **`@settings(max_examples=20)` trong nhiều test**: Số nhỏ vì test này mock LLM, không cần nhiều examples. Với test không mock (test thật), nên dùng 100+.

---

## 10. LIÊN HỆ TỔNG THỂ — Tất cả kết nối như thế nào

### 10.1 Luồng dữ liệu đầy đủ — Từ input đến output

```
INPUT: "Khách chê Dell Inspiron đắt"
│
├─► SalesResearchAgent.run("Khách chê Dell Inspiron đắt")
│       │
│       ├─► query = "Khách chê Dell Inspiron đắt"  (không có correction)
│       │
│       └─► self._agent.chat(query)
│               │
│               └─► ReAct Loop (LlamaIndex)
│                       │
│                       ├─► Thought: "Cần tìm thông tin Dell Inspiron"
│                       │
│                       ├─► Action: internal_db_search("Dell Inspiron thông số giá")
│                       │       │
│                       │       ├─► cache.get("Dell Inspiron...") → None (miss)
│                       │       │
│                       │       ├─► rag_pipeline.query("Dell Inspiron...")
│                       │       │       │
│                       │       │       ├─► relevance_checker.check() → "CAN_ANSWER"
│                       │       │       │
│                       │       │       └─► hybrid_retriever.retrieve()
│                       │       │               │
│                       │       │               ├─► BM25 search (parallel)
│                       │       │               ├─► Vector search (parallel)
│                       │       │               ├─► RRF fusion
│                       │       │               └─► [NodeWithScore, ...]
│                       │       │
│                       │       ├─► Format JSON: [{source, product_code, content[:500]}]
│                       │       ├─► cache.set("Dell Inspiron...", json_result)
│                       │       └─► return json_string
│                       │
│                       ├─► Observation: [{"source": "product_catalog", "content": "Dell Inspiron..."}]
│                       │
│                       ├─► Thought: "Đã có đủ thông tin, không cần web search"
│                       │
│                       └─► Final Answer: "Dạ, em hiểu anh/chị đang so sánh về giá..."
│
└─► AgentResult(
        objection_text="Khách chê Dell Inspiron đắt",
        draft_response="Dạ, em hiểu anh/chị...",
        tools_used=["internal_db_search"]
    )
```

---

### 10.2 Các Design Decisions quan trọng và lý do

| Quyết định | Lý do | Hệ quả |
|---|---|---|
| `max_iterations=2` | Kiểm soát chi phí + latency | Tối đa 2 tool calls, response < 10s |
| Tool description có Micro-prompt | LLM đọc description để quyết định | Smart routing không cần code logic |
| `isinstance(result, str)` check | RAGPipeline trả về 2 kiểu khác nhau | Xử lý NO_MATCH đúng cách |
| `ensure_ascii=False` | Dữ liệu tiếng Việt | LLM đọc được Unicode |
| `field(default_factory=list)` | Tránh shared mutable default | Mỗi AgentResult có list riêng |
| `TYPE_CHECKING` import | Tránh circular import | Module độc lập, dễ test |
| `except Exception` trong run() | Never crash | Property 8 luôn đúng |
| `objection_text=objection` (không phải query) | Round-trip property | Property 9 luôn đúng |
| Factory pattern cho tools | Closure over dependencies | Dễ inject mock trong test |
| OrderedDict cho cache | O(1) LRU eviction | Performance tốt |

---

### 10.3 Correction Loop — Cách hệ thống tự sửa lỗi

```
Lần 1:
  run("iPhone quá đắt") → AgentResult(draft="Dạ, iPhone 15 giá 35M...")
                                              ↓
                              Verification Agent kiểm tra
                                              ↓
                              "Sai! Giá thực tế là 30M, sai lệch 16.7%"
                                              ↓
Lần 2 (retry):
  run("iPhone quá đắt",
      correction_feedback="Correct the price",
      verification_issues=[PriceIssue(product="iPhone 15", mentioned="35M", actual="30M")])
                                              ↓
  query = "⚠️ Đây là lần thử lại...
           • [GIÁ] iPhone 15: đề cập '35M', thực tế '30M' (sai lệch 16.7%)
           ---
           CÂU HỎI GỐC: iPhone quá đắt"
                                              ↓
  AgentResult(draft="Dạ, iPhone 15 có giá 30M...")  ← Đã sửa
```

---

## 11. BÀI HỌC RÚT RA — Patterns quan trọng khi xây Agent

### Pattern 1: ReAct = Thought + Action + Observation

Đây là pattern cốt lõi của mọi LLM Agent hiện đại:

```
Thought:     LLM suy luận về bước tiếp theo
Action:      Gọi tool với input cụ thể
Observation: Đọc output của tool
→ Lặp lại cho đến khi có Final Answer
```

LlamaIndex tự động quản lý vòng lặp này. Bạn chỉ cần:
1. Định nghĩa tools (FunctionTool)
2. Viết System Prompt hướng dẫn khi nào dùng tool nào
3. Set `max_iterations` để kiểm soát

---

### Pattern 2: Tool Description = Micro-Prompt

Description của tool không chỉ là documentation — LLM đọc nó để quyết định có gọi tool không và khi nào gọi. Viết description như viết instruction:

```python
# Tệ:
description="Search the database"

# Tốt:
description=(
    "Tra cứu thông tin sản phẩm, giá, thông số kỹ thuật và chính sách "
    "bảo hành/đổi trả từ cơ sở dữ liệu nội bộ của công ty. "
    "Luôn gọi tool này TRƯỚC khi dùng web search."
)
```

---

### Pattern 3: Output Contract = Dataclass

Luôn định nghĩa output của Agent là một dataclass rõ ràng:
- Dễ serialize (`dataclasses.asdict()`)
- Dễ test (kiểm tra từng field)
- Dễ extend (thêm optional fields mà không break backward compatibility)
- Type-safe

---

### Pattern 4: Never Crash = try-except + Fallback

Agent chạy trong production phải **không bao giờ crash**:

```python
try:
    response = self._agent.chat(query)
    return AgentResult(...)
except Exception as exc:
    logger.error("Agent error: %s", exc, exc_info=True)
    return AgentResult(
        objection_text=objection,
        draft_response="Dạ, hệ thống đang gặp sự cố...",  # ← Fallback an toàn
        tools_used=[],
    )
```

---

### Pattern 5: Memory Safety = Truncate + JSON

Khi truyền dữ liệu từ tool vào LLM context:
1. **Truncate** nội dung dài (500 ký tự)
2. **JSON format** để LLM parse dễ hơn
3. **Chỉ giữ fields cần thiết** (source, product_code, content)

---

### Pattern 6: Smart Routing = Prompt Engineering + Tool Design

Để Agent luôn ưu tiên DB nội bộ trước web:
1. System Prompt: "LUÔN gọi internal_db_search TRƯỚC TIÊN"
2. Tool description: "Luôn gọi tool này TRƯỚC khi dùng web search"
3. Tool description của Tavily: "CHỈ dùng khi Internal_DB_Tool không có thông tin"

Ba lớp enforcement → LLM rất khó "quên" rule này.

---

### Pattern 7: Property-Based Testing cho Agent

Thay vì test với 5-10 cases cụ thể, định nghĩa **tính chất bất biến**:
- "Với MỌI input, output không bao giờ crash"
- "Với MỌI input, objection_text == input gốc"
- "Với MỌI input, len(tools_used) <= 2"

Hypothesis tự tìm counterexample → phát hiện bugs mà bạn không nghĩ tới.

---

### Pattern 8: Hybrid Retrieval = BM25 + Vector + RRF

- **BM25**: Tốt cho exact match (SKU, tên sản phẩm cụ thể)
- **Vector**: Tốt cho semantic similarity (câu hỏi diễn đạt khác nhau)
- **RRF**: Gộp kết quả bằng rank (không phải score) → chuẩn hóa được
- **SKU Boost**: Business rule đặc biệt — nếu query chứa SKU, node đó phải lên đầu

---

### Pattern 9: Correction Loop = Structured Feedback

Khi Agent cần retry:
1. Không chỉ nói "sai rồi" — nói **chính xác** sai ở đâu
2. Format feedback thành structured data (PriceIssue, PolicyIssue, RelevanceIssue)
3. Prepend vào query để Agent biết cần sửa gì

---

### Pattern 10: Conditional Tool Registration

```python
tavily_tool = build_tavily_tool(tavily_api_key)
if tavily_tool is not None:
    tools.append(tavily_tool)
```

Tool tùy chọn → trả về `None` nếu không có điều kiện (API key). Hệ thống vẫn hoạt động với chỉ Internal DB tool.

---

## TÓM TẮT CUỐI — Bức tranh hoàn chỉnh

```
[Nhân viên] → Objection → [SalesResearchAgent]
                                    │
                          ┌─────────┴─────────┐
                          │   ReAct Loop       │
                          │  max_iterations=2  │
                          └─────────┬─────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              [Internal DB Tool]            [Tavily Tool]
              (luôn gọi trước)              (chỉ khi cần)
                    │                               │
              [RAGPipeline]                  [Tavily API]
                    │
          ┌─────────┴─────────┐
          │                   │
    [RelevanceChecker]  [HybridRetriever]
    (lọc query)         (BM25 + Vector + RRF)
                                    │
                              [ChromaDB + docstore.json]
                                    │
                              [AgentResult]
                                    │
                          [Verification Agent] (Tuần 6)
```

**Mỗi layer giải quyết một vấn đề cụ thể:**
- RelevanceChecker → Tránh tìm kiếm vô ích
- HybridRetriever → Tìm kiếm chính xác và ngữ nghĩa
- RAGPipeline → Điều phối 2 component trên
- Internal DB Tool → Wrap RAGPipeline + Memory Safety + Caching
- Tavily Tool → Fallback khi DB không có
- Agent Prompt → Điều khiển hành vi Agent bằng ngôn ngữ
- SalesResearchAgent → Orchestrate tất cả + Error handling
- AgentResult → Output contract chuẩn hóa
- QueryCache → Performance optimization
- PBT → Đảm bảo correctness properties với mọi input
