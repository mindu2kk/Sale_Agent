# Giải thích chi tiết: Hybrid Retriever & Relevance Checker

> Tài liệu này giải thích toàn bộ công việc đã hoàn thành trong spec `hybrid-retriever-relevance-checker`.
> Mục tiêu: giúp người mới đọc code hiểu **từng file làm gì**, **code nào ở đâu**, và **tại sao lại làm vậy**.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Task 1 — Cấu trúc project & dependencies](#2-task-1--cấu-trúc-project--dependencies)
3. [Task 2 — HybridRetriever: khởi tạo](#3-task-2--hybridretriever-khởi-tạo)
4. [Task 3 — HybridRetriever: retrieve()](#4-task-3--hybridretriever-retrieve)
5. [Task 4 — RelevanceChecker](#5-task-4--relevancechecker)
6. [Task 5 — RAGPipeline](#6-task-5--ragpipeline)
7. [Task 6 — Unit Tests](#7-task-6--unit-tests)
8. [Task 7 — Property-Based Tests (Hypothesis)](#8-task-7--property-based-tests-hypothesis)
9. [Task 8 — Integration & Performance Tests](#9-task-8--integration--performance-tests)
10. [Luồng dữ liệu end-to-end](#10-luồng-dữ-liệu-end-to-end)

---

## 1. Tổng quan kiến trúc

Hệ thống này là một **RAG pipeline** (Retrieval-Augmented Generation) cho chatbot tư vấn bán hàng điện tử. Khi người dùng đặt câu hỏi, hệ thống:

1. **Phân loại câu hỏi** (RelevanceChecker) — xem câu hỏi có liên quan đến sản phẩm/chính sách không.
2. **Tìm kiếm tài liệu** (HybridRetriever) — kết hợp BM25 (keyword) + ChromaDB (vector) để tìm nodes liên quan nhất.
3. **Trả kết quả** (RAGPipeline) — routing: nếu không liên quan thì trả lời mặc định, nếu liên quan thì trả về danh sách nodes.

```
User Query
    │
    ▼
RelevanceChecker.check()
    │
    ├── NO_MATCH ──────────────────► "Xin lỗi, câu hỏi nằm ngoài phạm vi..."
    │
    └── CAN_ANSWER / PARTIAL
            │
            ▼
    HybridRetriever.retrieve()
            │
            ├── BM25 search (keyword)  ─┐
            │                           ├── RRF Fusion ──► SKU Boost ──► top_k nodes
            └── Vector search (ChromaDB)┘
```

---

## 2. Task 1 — Cấu trúc project & dependencies

### File: `retriever/__init__.py`

```python
from retriever.hybrid_retriever import HybridRetriever
from retriever.relevance_checker import RelevanceChecker

__all__ = ["HybridRetriever", "RelevanceChecker"]
```

**Nhiệm vụ:** Export cả hai class ra ngoài để code khác chỉ cần `from retriever import HybridRetriever, RelevanceChecker` mà không cần biết file nội bộ.

### File: `requirements.txt`

Thêm hai dependency quan trọng:
- `rank-bm25` — thư viện BM25Okapi để tìm kiếm keyword
- `hypothesis` — thư viện property-based testing

---

## 3. Task 2 — HybridRetriever: khởi tạo

### File: `retriever/hybrid_retriever.py` — class `HybridRetriever.__init__()`

```python
def __init__(
    self,
    docstore_path: str = "./chroma_db/docstore.json",
    chroma_path: str = "./chroma_db",
    collection_name: str = "sales_copilot_vdb",
    embed_model_name: str = "BAAI/bge-m3",
    bm25_top_k: int = 20,
    vector_top_k: int = 20,
    rrf_k: int = 60,
) -> None:
```

Constructor nhận các tham số cấu hình và gọi hai hàm khởi tạo con:

---

### `_init_bm25()` — Task 2.1, 2.2, 2.4

```python
def _init_bm25(self) -> None:
    if not os.path.exists(self.docstore_path):
        raise FileNotFoundError(...)   # Task 2.4: lỗi rõ ràng nếu file không tồn tại

    with open(self.docstore_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docstore_data = data.get("docstore/data", {})
    nodes = []
    for node_id, node_entry in docstore_data.items():
        node_data = node_entry.get("__data__", {})
        nodes.append(TextNode(**node_data))   # Task 2.1: load nodes từ JSON

    self._nodes = nodes
    self._bm25_corpus = [node.text.split() for node in self._nodes]  # tokenize

    if self._bm25_corpus:
        self._bm25 = BM25Okapi(self._bm25_corpus)   # Task 2.2: build BM25 index
    else:
        self._bm25 = BM25Okapi([[""]])  # Task 2.4: corpus rỗng không crash
```

**Giải thích:**
- `docstore.json` là file JSON chứa tất cả các "nodes" (đoạn văn bản) đã được ingestion pipeline tạo ra.
- Mỗi node có `id_`, `text`, và `metadata` (ví dụ: `source_type`, `product_code`).
- BM25Okapi cần corpus được tokenize (split theo whitespace). Nếu corpus rỗng, dùng dummy `[[""]]` để tránh crash.

---

### `_init_vector()` — Task 2.3, 2.4

```python
def _init_vector(self) -> None:
    db = chromadb.PersistentClient(path=self.chroma_path)

    existing = [c.name for c in db.list_collections()]
    if self.collection_name not in existing:
        raise ValueError(...)   # Task 2.4: lỗi rõ ràng nếu collection không tồn tại

    chroma_collection = db.get_collection(self.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
    self._vector_retriever = index.as_retriever(
        similarity_top_k=self.vector_top_k,
        embed_model=embed_model,
    )
```

**Giải thích:**
- Kết nối ChromaDB (vector database lưu trên disk tại `./chroma_db`).
- Kiểm tra collection tồn tại trước khi dùng — nếu không có thì raise `ValueError` với message rõ ràng.
- Tạo `VectorStoreIndex` từ ChromaDB và wrap thành retriever dùng embedding model `BAAI/bge-m3`.

---

## 4. Task 3 — HybridRetriever: retrieve()

### File: `retriever/hybrid_retriever.py` — method `retrieve()`

Đây là phương thức chính, thực hiện 5 bước:

---

### Bước 1: Parallel search — Task 3.1

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_bm25 = executor.submit(self._bm25_search, query)
    future_vector = executor.submit(self._vector_search, query)
    bm25_results = future_bm25.result()
    vector_results = future_vector.result()
```

**Tại sao:** BM25 và vector search độc lập nhau. Chạy song song bằng `ThreadPoolExecutor` giảm latency tổng thể (thay vì chạy tuần tự).

---

### Bước 2: RRF Fusion — Task 3.2

```python
fusion_scores: dict[str, float] = {}
node_map: dict[str, NodeWithScore] = {}

for result_list in (bm25_results, vector_results):
    for rank, nws in enumerate(result_list, start=1):
        node_id = nws.node.node_id
        fusion_scores[node_id] = fusion_scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
        if node_id not in node_map:
            node_map[node_id] = nws
```

**Công thức RRF:** `fusion_score += 1 / (k + rank)` với `k = 60` (hằng số chuẩn trong literature).

**Ví dụ cụ thể:**
- Node A: rank 1 trong BM25, rank 2 trong vector → `1/61 + 1/62 ≈ 0.0325`
- Node B: rank 2 trong BM25 only → `1/62 ≈ 0.0161`
- → Node A được xếp trên Node B

**Tại sao RRF:** Kết hợp hai nguồn tìm kiếm khác nhau (keyword vs semantic) mà không cần normalize score — chỉ dùng rank.

---

### Bước 3: Sort + Tie-breaking — Task 3.3, 3.6

```python
source_priority = {"product_catalog": 1, "policy_pdf": 0}

def sort_key(node_id: str):
    score = fusion_scores[node_id]
    priority = source_priority.get(
        node_map[node_id].node.metadata.get("source_type", ""), 0
    )
    return (score, priority)

sorted_ids = sorted(fusion_scores.keys(), key=sort_key, reverse=True)
```

**Tie-breaking:** Khi hai node có cùng fusion score, node có `source_type = "product_catalog"` được ưu tiên hơn `"policy_pdf"`. Lý do: thông tin sản phẩm cụ thể thường hữu ích hơn chính sách chung.

---

### Bước 4: top_k — Task 3.4

```python
results = results[:top_k]
```

Đơn giản cắt danh sách sau khi sort.

---

### Bước 5: SKU Boost — Task 3.5

```python
query_tokens = query.split()
boost_index = None
for i, nws in enumerate(results):
    product_code = nws.node.metadata.get("product_code", "")
    if product_code and product_code in query_tokens:
        boost_index = i
        break

if boost_index is not None and boost_index != 0:
    boosted = results.pop(boost_index)
    results.insert(0, boosted)
```

**Tại sao:** Nếu người dùng gõ đúng mã SKU (ví dụ `IP15-PRM-256`), node tương ứng phải xuất hiện ở vị trí đầu tiên — dù RRF score của nó thấp hơn. Đây là business rule quan trọng cho hệ thống bán hàng.

---

## 5. Task 4 — RelevanceChecker

### File: `retriever/relevance_checker.py` — class `RelevanceChecker`

```python
_SYSTEM_PROMPT_TEMPLATE = """Bạn là bộ phân loại câu hỏi cho hệ thống tư vấn bán hàng điện tử.
Phân loại câu hỏi sau vào đúng một trong ba nhãn:
- CAN_ANSWER: câu hỏi liên quan trực tiếp đến sản phẩm, mã SKU, giá, hoặc chính sách bảo hành/đổi trả
- PARTIAL: câu hỏi liên quan một phần (thương hiệu chung, câu hỏi mơ hồ có thể liên quan)
- NO_MATCH: câu hỏi hoàn toàn không liên quan đến sản phẩm hoặc chính sách

Chỉ trả về đúng một nhãn, không giải thích thêm.

Câu hỏi: {query}
Nhãn:"""
```

**Nhiệm vụ:** Prompt này được fix cứng trong `__init__` (Task 4.1). Nó hướng dẫn LLM chỉ trả về một trong ba nhãn.

---

### `check()` — Task 4.2, 4.3

```python
def check(self, query: str) -> RelevanceLabel:
    if query.strip() == "":
        return "NO_MATCH"

    prompt = self._prompt_template.format(query=query)
    response = self.llm.complete(prompt)
    raw = response.text.strip().rstrip(".!?").upper()   # Task 4.3: normalize

    valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
    if raw in valid_labels:
        return raw
    return "PARTIAL"   # Task 4.3: fallback nếu LLM trả về gì đó lạ
```

**Label parsing (Task 4.3):**
- `.strip()` — xóa whitespace đầu/cuối
- `.rstrip(".!?")` — xóa dấu chấm/chấm than/hỏi thừa
- `.upper()` — chuyển về chữ hoa để so sánh
- Nếu không khớp → fallback về `"PARTIAL"` (an toàn nhất, vẫn gọi retriever)

---

## 6. Task 5 — RAGPipeline

### File: `rag_pipeline.py` — class `RAGPipeline`

```python
class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, checker: RelevanceChecker) -> None:
        self.retriever = retriever
        self.checker = checker

    def query(self, user_query: str) -> Union[str, list[NodeWithScore]]:
        label = self.checker.check(user_query)          # Task 5.2: gọi checker trước
        logger.info(f"[RelevanceChecker] label={label} query={user_query!r}")  # Task 5.2: log

        if label == "NO_MATCH":
            return _DEFAULT_NO_MATCH_RESPONSE           # Task 5.3: routing NO_MATCH

        return self.retriever.retrieve(user_query)      # Task 5.3: routing CAN_ANSWER/PARTIAL
```

**Routing logic (Task 5.3):**
- `NO_MATCH` → trả về string mặc định, **không gọi retriever** (tiết kiệm API call)
- `CAN_ANSWER` hoặc `PARTIAL` → gọi `retriever.retrieve()` và trả về list nodes

**Default response:**
```python
_DEFAULT_NO_MATCH_RESPONSE = (
    "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của hệ thống. "
    "Vui lòng hỏi về sản phẩm hoặc chính sách bảo hành/đổi trả."
)
```

---

## 7. Task 6 — Unit Tests

### File: `tests/test_unit.py`

Tất cả unit tests đều **mock** các external services (ChromaDB, LLM, embedding model) để chạy nhanh và không cần infrastructure thật.

| Test class / function | Task | Kiểm tra gì |
|---|---|---|
| `TestDocstoreNotFound` | 6.1 | `FileNotFoundError` khi `docstore_path` sai |
| `TestCollectionNotFound` | 6.2 | `ValueError` khi ChromaDB collection không tồn tại |
| `TestRRFFormula` | 6.3 | Công thức RRF tính đúng: `1/(k+rank)` |
| `TestSKUBoost` | 6.4 | Node có `product_code` khớp query xuất hiện ở index 0 |
| `TestTieBreaking` | 6.5 | `product_catalog` xếp trước `policy_pdf` khi cùng score |
| `TestRelevanceCheckerLabels` | 6.6 | Mock LLM trả về `CAN_ANSWER`, `PARTIAL`, `NO_MATCH` đều được parse đúng |
| `TestLabelParsing` | 6.7 | `"  CAN_ANSWER. "` → `"CAN_ANSWER"`, `"no_match."` → `"NO_MATCH"` |
| `TestNoMatchSkipsRetrieval` | 6.8 | `retriever.retrieve` **không được gọi** khi label là `NO_MATCH` |
| `TestEmptyCorpus` | 6.9 | Corpus rỗng không crash, `retrieve()` trả về `[]` |

**Kỹ thuật mock điển hình:**

```python
# Mock _init_vector để không cần ChromaDB thật
with patch.object(HybridRetriever, "_init_vector", return_value=None):
    retriever = HybridRetriever(docstore_path=tmp_path, ...)

# Mock LLM response
mock_response = MagicMock()
mock_response.text = "CAN_ANSWER"
mock_llm = MagicMock()
mock_llm.complete.return_value = mock_response
checker = RelevanceChecker(llm=mock_llm)
```

---

## 8. Task 7 — Property-Based Tests (Hypothesis)

### File: `tests/test_pbt.py`

Property-based testing dùng thư viện **Hypothesis** — tự động sinh hàng trăm input ngẫu nhiên để kiểm tra các **bất biến** (properties) của hệ thống.

| Test function | Task | Property được kiểm tra |
|---|---|---|
| `test_bm25_roundtrip` | 7.1 | Với bất kỳ node nào trong corpus, query bằng chính text của node đó phải trả về node đó trong top-5 BM25 |
| `test_hybrid_retriever_idempotence` | 7.2 | Hai lần khởi tạo từ cùng docstore → cùng thứ tự kết quả cho cùng query |
| `test_topk_bound` | 7.3 | `len(results) <= top_k` với mọi query và mọi giá trị top_k |
| `test_sku_exact_match_top3` | 7.4 | Node có `product_code` khớp query luôn nằm trong top-3 |
| `test_rrf_score_monotonicity` | 7.5 | Node có rank tốt hơn ở cả hai retriever → fusion score cao hơn |
| `test_relevance_label_validity` | 7.6 | `check()` luôn trả về một trong `{"CAN_ANSWER", "PARTIAL", "NO_MATCH"}` |
| `test_no_match_skips_retrieval` | 7.7 | `retriever.retrieve` không bao giờ được gọi khi checker trả về `NO_MATCH` |
| `test_union_pool_completeness` | 7.8 | Mọi node từ BM25 hoặc vector search đều xuất hiện trong fusion pool |

**Ví dụ property test:**

```python
@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_rrf_score_monotonicity(rank_a: int, rank_b: int) -> None:
    k = 60
    score_a = 1.0 / (k + rank_a) + 1.0 / (k + rank_a)
    score_b = 1.0 / (k + rank_b) + 1.0 / (k + rank_b)

    if rank_a < rank_b:
        assert score_a > score_b  # rank tốt hơn → score cao hơn
```

Hypothesis sẽ tự sinh 100 cặp `(rank_a, rank_b)` ngẫu nhiên và kiểm tra property này với tất cả.

---

## 9. Task 8 — Integration & Performance Tests

### File: `tests/test_integration.py`

Các test này dùng **infrastructure thật** (docstore.json, ChromaDB trên disk) và LLM thật (Google Gemini).

---

### Task 8.1 — End-to-end test

```python
class TestEndToEnd:
    PRODUCT_QUERIES = [
        "iPhone 15 Pro Max giá bao nhiêu?",
        "Chính sách bảo hành Samsung là gì?",
        "Cho tôi xem thông tin sản phẩm IP15-PRM-256",
    ]
```

**Kiểm tra:**
- Query sản phẩm → trả về list `NodeWithScore` với metadata hợp lệ (`source_type` hoặc `file_name`)
- Query off-topic → trả về `_DEFAULT_NO_MATCH_RESPONSE`
- Mọi node trả về đều có `text` không rỗng
- `top_k` được tôn trọng trong retrieval thật

---

### Task 8.2 — HybridRetriever latency ≤ 3s

```python
def test_retriever_latency_under_3s(self, hybrid_retriever):
    for query in self.PERF_QUERIES:
        start = time.perf_counter()
        results = hybrid_retriever.retrieve(query, top_k=10)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Too slow: {elapsed:.3f}s"
```

**Tại sao 3s:** Requirement 7.1 trong spec. Parallel execution (ThreadPoolExecutor) giúp đạt được ngưỡng này.

---

### Task 8.3 — RelevanceChecker latency ≤ 5s

```python
@requires_gemini
def test_checker_latency_under_5s(self, relevance_checker):
    for query in self.PERF_QUERIES:
        start = time.perf_counter()
        label = relevance_checker.check(query)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Too slow: {elapsed:.3f}s"
```

**Decorator `@requires_gemini`:** Tự động skip test nếu `GOOGLE_API_KEY` không có hoặc quota Gemini đã hết. Điều này giúp CI/CD không fail khi không có API key.

```python
_GEMINI_AVAILABLE, _GEMINI_SKIP_REASON = _check_gemini_available()
requires_gemini = pytest.mark.skipif(
    not _GEMINI_AVAILABLE,
    reason=_GEMINI_SKIP_REASON or "Gemini API unavailable",
)
```

---

## 10. Luồng dữ liệu end-to-end

Dưới đây là ví dụ cụ thể với query `"IP15-PRM-256 iPhone 15 Pro Max giá bao nhiêu?"`:

```
1. RAGPipeline.query("IP15-PRM-256 iPhone 15 Pro Max giá bao nhiêu?")
   │
   ▼
2. RelevanceChecker.check(query)
   │  → LLM nhận prompt với query
   │  → LLM trả về "CAN_ANSWER"
   │  → parse: strip + upper → "CAN_ANSWER" ✓
   │
   ▼
3. HybridRetriever.retrieve(query, top_k=10)
   │
   ├── ThreadPoolExecutor:
   │   ├── BM25: tokenize query → ["IP15-PRM-256", "iPhone", "15", "Pro", "Max", "giá", "bao", "nhiêu"]
   │   │         → BM25Okapi.get_scores() → top 20 nodes theo keyword match
   │   │
   │   └── Vector: embed query → cosine similarity → top 20 nodes từ ChromaDB
   │
   ├── RRF Fusion:
   │   ├── Node "ip15-node" (rank 3 BM25, rank 2 vector): score = 1/63 + 1/62 ≈ 0.0320
   │   ├── Node "samsung-node" (rank 1 BM25 only):        score = 1/61 ≈ 0.0164
   │   └── ... (các nodes khác)
   │
   ├── Sort giảm dần theo score → ["ip15-node", "samsung-node", ...]
   │
   ├── top_k=10 → cắt còn 10 nodes
   │
   └── SKU Boost: "IP15-PRM-256" có trong query tokens
                  → tìm node có product_code="IP15-PRM-256"
                  → đưa lên index 0 (dù score thấp hơn)
   │
   ▼
4. Trả về: [NodeWithScore(node=ip15-node, score=...), ...]
```

---

## Tóm tắt các file và nhiệm vụ

| File | Nhiệm vụ chính |
|---|---|
| `retriever/hybrid_retriever.py` | Class `HybridRetriever`: load docstore → build BM25 → kết nối ChromaDB → retrieve với RRF + SKU boost |
| `retriever/relevance_checker.py` | Class `RelevanceChecker`: gọi LLM phân loại query thành CAN_ANSWER / PARTIAL / NO_MATCH |
| `retriever/__init__.py` | Export `HybridRetriever` và `RelevanceChecker` |
| `rag_pipeline.py` | Class `RAGPipeline`: orchestrate checker → routing → retriever |
| `tests/test_unit.py` | 9 nhóm unit test với mock, kiểm tra từng behavior cụ thể |
| `tests/test_pbt.py` | 8 property-based tests với Hypothesis, kiểm tra bất biến hệ thống |
| `tests/test_integration.py` | End-to-end tests với infrastructure thật + performance tests (latency ≤ 3s/5s) |

---

## Các khái niệm kỹ thuật quan trọng

### BM25 (Best Match 25)
Thuật toán tìm kiếm keyword cổ điển, tính điểm dựa trên tần suất từ (TF) và nghịch đảo tần suất tài liệu (IDF). Tốt cho các query có từ khóa chính xác như mã SKU.

### Vector Search (ChromaDB)
Tìm kiếm ngữ nghĩa: embed query và documents thành vector, tìm các vector gần nhau nhất (cosine similarity). Tốt cho các query mô tả ý nghĩa, không cần từ khóa chính xác.

### RRF (Reciprocal Rank Fusion)
Phương pháp kết hợp kết quả từ nhiều retriever: `score = Σ 1/(k + rank)`. Không cần normalize score từ các nguồn khác nhau, chỉ dùng thứ hạng. `k=60` là giá trị chuẩn trong literature.

### Property-Based Testing
Thay vì viết test case cụ thể, định nghĩa **property** (bất biến) và để Hypothesis tự sinh input ngẫu nhiên để tìm counterexample. Ví dụ: "với mọi top_k, kết quả không bao giờ dài hơn top_k".



---

# Giải thích hệ thống theo batch — Từ tổng thể đến chi tiết

---

## Batch 1 — Hệ thống này làm gì? Tại sao lại cần?

### Vấn đề cần giải quyết

Hãy tưởng tượng bạn có một chatbot tư vấn bán hàng điện tử. Khách hàng gõ vào:

> "iPhone 15 Pro Max 256GB giá bao nhiêu?"

Chatbot cần trả lời đúng, nhanh, và không bịa đặt. Để làm được điều đó, nó cần **tìm đúng thông tin** từ kho dữ liệu (catalog sản phẩm, chính sách bảo hành...) trước khi trả lời.

Đây là bài toán **RAG — Retrieval-Augmented Generation**: thay vì để LLM tự bịa, ta *lấy* tài liệu liên quan trước, rồi mới để LLM trả lời dựa trên tài liệu đó.

---

### Ba thành phần cốt lõi

```
User Query
    │
    ▼
[1] RelevanceChecker   ← "Câu hỏi này có liên quan không?"
    │
    ├── KHÔNG liên quan → trả lời mặc định, DỪNG
    │
    └── CÓ liên quan
            │
            ▼
[2] HybridRetriever    ← "Tìm tài liệu liên quan nhất"
            │
            ▼
[3] RAGPipeline        ← "Điều phối toàn bộ luồng trên"
```

**Tại sao cần RelevanceChecker trước?**
Nếu khách hỏi "hôm nay thời tiết thế nào?" — không cần tìm kiếm gì cả, tiết kiệm tài nguyên, trả lời ngay "ngoài phạm vi hỗ trợ".

**Tại sao cần HybridRetriever thay vì chỉ dùng một loại tìm kiếm?**
- Tìm kiếm **keyword (BM25)**: giỏi với mã SKU chính xác như `IP15-PRM-256`
- Tìm kiếm **ngữ nghĩa (Vector)**: giỏi với câu hỏi mô tả như "điện thoại cao cấp của Apple"
- Kết hợp cả hai → tốt hơn cả hai

---

### Sơ đồ file → nhiệm vụ

| File | Vai trò |
|---|---|
| `retriever/relevance_checker.py` | Thành phần [1] — hỏi LLM xem query có liên quan không |
| `retriever/hybrid_retriever.py` | Thành phần [2] — tìm kiếm kết hợp BM25 + Vector |
| `rag_pipeline.py` | Thành phần [3] — điều phối, routing |
| `tests/test_unit.py` | Kiểm tra từng hành vi nhỏ (có mock) |
| `tests/test_pbt.py` | Kiểm tra bất biến hệ thống (Hypothesis) |
| `tests/test_integration.py` | Kiểm tra end-to-end với hạ tầng thật |

---

### Điểm mấu chốt cần nhớ từ Batch 1

> Hệ thống hoạt động theo **pipeline tuyến tính**: Checker → Retriever → Kết quả. Mỗi bước có thể "short-circuit" (dừng sớm) nếu không cần thiết. Đây là pattern thiết kế quan trọng — tiết kiệm tài nguyên và giữ code rõ ràng.


---

## Batch 2 — RelevanceChecker: Cửa ngõ của hệ thống

### Vị trí trong pipeline

```
User Query → [RelevanceChecker] → HybridRetriever → Kết quả
                    ↑
              Batch 2 phân tích phần này
```

RelevanceChecker là thành phần **đầu tiên** được gọi. Nó quyết định có cần tìm kiếm gì không. Nếu không cần → dừng ngay, tiết kiệm toàn bộ chi phí gọi retriever và embedding model.

---

### Kiến trúc của class

```python
# File: retriever/relevance_checker.py

RelevanceLabel = Literal["CAN_ANSWER", "PARTIAL", "NO_MATCH"]
```

Đây là **type alias** — Python sẽ kiểm tra tĩnh rằng hàm `check()` chỉ được phép trả về đúng một trong ba chuỗi này. Nếu bạn viết `return "MAYBE"` thì IDE/mypy sẽ báo lỗi ngay. Đây là cách dùng type system để enforce business rule.

---

### Prompt template — trái tim của RelevanceChecker

```python
_SYSTEM_PROMPT_TEMPLATE = """Bạn là bộ phân loại câu hỏi cho hệ thống tư vấn bán hàng điện tử.
Phân loại câu hỏi sau vào đúng một trong ba nhãn:
- CAN_ANSWER: câu hỏi liên quan trực tiếp đến sản phẩm, mã SKU, giá, hoặc chính sách bảo hành/đổi trả
- PARTIAL: câu hỏi liên quan một phần (thương hiệu chung, câu hỏi mơ hồ có thể liên quan)
- NO_MATCH: câu hỏi hoàn toàn không liên quan đến sản phẩm hoặc chính sách

Chỉ trả về đúng một nhãn, không giải thích thêm.

Câu hỏi: {query}
Nhãn:"""
```

Phân tích từng phần của prompt này:

**"Chỉ trả về đúng một nhãn, không giải thích thêm"** — đây là kỹ thuật prompt engineering quan trọng. LLM có xu hướng giải thích dài dòng. Câu lệnh này ép LLM trả về output ngắn gọn, dễ parse. Nếu không có câu này, LLM có thể trả về `"CAN_ANSWER vì câu hỏi đề cập đến giá sản phẩm..."` — và code parse sẽ fail.

**`{query}`** — placeholder được điền bằng `.format(query=query)` trong `check()`. Đây là string template chuẩn của Python, không phải f-string, vì template được định nghĩa ở module level (trước khi có giá trị query).

**Ba nhãn với định nghĩa rõ ràng** — mỗi nhãn có ví dụ cụ thể trong mô tả. Điều này giúp LLM phân biệt ranh giới giữa `CAN_ANSWER` và `PARTIAL`. Ví dụ: "Samsung có tốt không?" → `PARTIAL` (thương hiệu chung), còn "Samsung Galaxy S24 giá bao nhiêu?" → `CAN_ANSWER` (sản phẩm cụ thể).

---

### `__init__()` — Dependency Injection

```python
def __init__(self, llm) -> None:
    self.llm = llm
    self._prompt_template = _SYSTEM_PROMPT_TEMPLATE
```

Điểm quan trọng: `llm` được **inject từ ngoài vào**, không được tạo bên trong class. Đây là pattern **Dependency Injection**.

Lợi ích thực tế:
- Trong production: truyền vào `Gemini` hoặc `OpenAI`
- Trong unit test: truyền vào `MagicMock()` — không cần API key, không tốn tiền, chạy nhanh

Nếu class tự tạo LLM bên trong `__init__`, bạn không thể test mà không có API key thật.

---

### `check()` — Luồng xử lý chi tiết

```python
def check(self, query: str) -> RelevanceLabel:
    # Bước 1: Guard clause — xử lý edge case trước
    if query.strip() == "":
        return "NO_MATCH"

    # Bước 2: Format prompt với query thực tế
    prompt = self._prompt_template.format(query=query)

    # Bước 3: Gọi LLM
    response = self.llm.complete(prompt)

    # Bước 4: Normalize output của LLM
    raw = response.text.strip().rstrip(".!?").upper()

    # Bước 5: Validate và fallback
    valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
    if raw in valid_labels:
        return raw
    return "PARTIAL"
```

**Bước 1 — Guard clause:** Query rỗng (`""` hoặc `"   "`) trả về `NO_MATCH` ngay lập tức. Không gọi LLM, không tốn API call. Đây là pattern "fail fast" — xử lý input không hợp lệ ở đầu hàm, trước khi làm bất cứ điều gì.

**Bước 4 — Normalize:** Chuỗi xử lý `.strip().rstrip(".!?").upper()` giải quyết các trường hợp LLM trả về không chuẩn:

| LLM trả về | Sau strip | Sau rstrip | Sau upper | Kết quả |
|---|---|---|---|---|
| `"  CAN_ANSWER  "` | `"CAN_ANSWER"` | `"CAN_ANSWER"` | `"CAN_ANSWER"` | ✓ |
| `"no_match."` | `"no_match."` | `"no_match"` | `"NO_MATCH"` | ✓ |
| `"PARTIAL!"` | `"PARTIAL!"` | `"PARTIAL"` | `"PARTIAL"` | ✓ |
| `"can_answer?"` | `"can_answer?"` | `"can_answer"` | `"CAN_ANSWER"` | ✓ |

**Bước 5 — Fallback về PARTIAL:** Tại sao `PARTIAL` chứ không phải `NO_MATCH`?

Đây là quyết định thiết kế quan trọng về **risk tolerance**:
- Nếu fallback về `NO_MATCH` → hệ thống từ chối trả lời câu hỏi có thể liên quan → **false negative** → khách hàng không được hỗ trợ
- Nếu fallback về `PARTIAL` → hệ thống vẫn gọi retriever → tốn thêm tài nguyên nhưng không bỏ sót câu hỏi hợp lệ

Trong hệ thống bán hàng, bỏ sót câu hỏi của khách hàng tệ hơn tốn thêm một chút tài nguyên. Nên `PARTIAL` là lựa chọn an toàn hơn.

---

### Điểm mấu chốt cần nhớ từ Batch 2

> `RelevanceChecker` là một **thin wrapper** quanh LLM call. Toàn bộ "trí tuệ" nằm trong prompt template. Code Python chỉ làm 3 việc: guard clause, gọi LLM, normalize output. Sự đơn giản này là có chủ đích — dễ test, dễ thay LLM, dễ thay prompt.

---

## Batch 3 — HybridRetriever: Khởi tạo (\_init\_bm25 và \_init\_vector)

### Vị trí trong pipeline

```
User Query → RelevanceChecker → [HybridRetriever.__init__] → retrieve()
                                         ↑
                                   Batch 3 phân tích phần này
```

Trước khi `retrieve()` có thể chạy, `__init__` phải chuẩn bị hai thứ:
1. **BM25 index** — được build từ `docstore.json` trên disk
2. **Vector retriever** — được kết nối từ ChromaDB trên disk

---

### Constructor — Tham số cấu hình

```python
def __init__(
    self,
    docstore_path: str = "./chroma_db/docstore.json",
    chroma_path: str = "./chroma_db",
    collection_name: str = "sales_copilot_vdb",
    embed_model_name: str = "BAAI/bge-m3",
    bm25_top_k: int = 20,
    vector_top_k: int = 20,
    rrf_k: int = 60,
) -> None:
```

Tất cả tham số đều có **default value** — bạn có thể khởi tạo `HybridRetriever()` không cần truyền gì. Nhưng trong production, bạn nên truyền explicit để rõ ràng.

Hai tham số quan trọng nhất:
- `bm25_top_k=20` và `vector_top_k=20`: mỗi retriever lấy 20 kết quả, sau đó RRF fusion chọn ra `top_k` (mặc định 10). Lấy nhiều hơn cần thiết để RRF có đủ "nguyên liệu" để chọn.
- `rrf_k=60`: hằng số RRF, giải thích chi tiết ở Batch 4.

---

### `_init_bm25()` — Build BM25 index từ docstore.json

```python
def _init_bm25(self) -> None:
    if not os.path.exists(self.docstore_path):
        raise FileNotFoundError(
            f"Docstore file not found at '{self.docstore_path}'. "
            "Please run the ingestion pipeline first to generate the docstore."
        )
```

**Guard clause đầu tiên:** Kiểm tra file tồn tại trước khi đọc. Error message có hướng dẫn hành động ("Please run the ingestion pipeline first") — đây là best practice, giúp developer biết phải làm gì khi gặp lỗi.

```python
    with open(self.docstore_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docstore_data: dict = data.get("docstore/data", {})
```

`docstore.json` có cấu trúc lồng nhau. Key `"docstore/data"` chứa dict các nodes. Dùng `.get(..., {})` thay vì `data["docstore/data"]` để tránh `KeyError` nếu file có cấu trúc khác.

```python
    nodes: list[TextNode] = []
    for node_id, node_entry in docstore_data.items():
        node_data: dict = node_entry.get("__data__", {})
        nodes.append(TextNode(**node_data))
```

Mỗi entry trong docstore có dạng:
```json
{
  "node-id-123": {
    "__data__": {
      "id_": "node-id-123",
      "text": "iPhone 15 Pro Max 256GB giá 34.990.000đ...",
      "metadata": {"source_type": "product_catalog", "product_code": "IP15-PRM-256"}
    }
  }
}
```

`TextNode(**node_data)` dùng **unpacking** để tạo object từ dict — tương đương gọi `TextNode(id_="...", text="...", metadata={...})`. Đây là cách LlamaIndex serialize/deserialize nodes.

```python
    self._bm25_corpus = [node.text.split() for node in self._nodes]

    if self._bm25_corpus:
        self._bm25 = BM25Okapi(self._bm25_corpus)
    else:
        self._bm25 = BM25Okapi([[""]])
```

**Tokenization:** BM25 cần corpus được tokenize. `.split()` chia theo whitespace — đơn giản nhưng đủ dùng cho tiếng Việt và tiếng Anh trong context này (mã SKU, tên sản phẩm).

**Empty corpus guard:** `BM25Okapi([])` sẽ crash. Dùng `[[""]]` (list chứa một document rỗng) để tạo object hợp lệ. Khi query, nó trả về score 0 cho tất cả — hành vi đúng khi không có dữ liệu.

---

### `_init_vector()` — Kết nối ChromaDB

```python
def _init_vector(self) -> None:
    db = chromadb.PersistentClient(path=self.chroma_path)

    existing = [c.name for c in db.list_collections()]
    if self.collection_name not in existing:
        raise ValueError(
            f"ChromaDB collection '{self.collection_name}' does not exist at "
            f"'{self.chroma_path}'. Please run the ingestion pipeline first."
        )
```

**Tại sao kiểm tra collection tồn tại?** Nếu không kiểm tra, `db.get_collection(name)` sẽ raise exception với message khó hiểu từ ChromaDB. Kiểm tra trước và raise `ValueError` với message rõ ràng là defensive programming — bảo vệ developer khỏi lỗi khó debug.

```python
    chroma_collection = db.get_collection(self.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
    self._vector_retriever = index.as_retriever(
        similarity_top_k=self.vector_top_k,
        embed_model=embed_model,
    )
```

Chuỗi khởi tạo này có 4 bước:
1. `db.get_collection()` → lấy ChromaDB collection object
2. `ChromaVectorStore(...)` → wrap collection thành LlamaIndex-compatible store
3. `VectorStoreIndex.from_vector_store(...)` → tạo index từ store (không load toàn bộ vectors vào RAM)
4. `index.as_retriever(...)` → tạo retriever object với config `similarity_top_k` và `embed_model`

**`BAAI/bge-m3`** là embedding model đa ngôn ngữ, hỗ trợ tốt tiếng Việt. Khi query, model này embed câu hỏi thành vector 1024 chiều, rồi ChromaDB tìm các vectors gần nhất trong collection.

Phần 1 — Khởi tạo: Chuẩn bị trước khi tìm kiếm
Khi bạn tạo HybridRetriever(), nó làm 2 việc chuẩn bị:

Việc 1: Đọc file docstore.json và build BM25 index

def _init_bm25(self) -> None:
Hình dung docstore.json như một cuốn sách chứa tất cả thông tin sản phẩm và chính sách, được chia thành nhiều đoạn nhỏ (gọi là "nodes"). Mỗi node là một đoạn văn bản, ví dụ:

Node 1: "iPhone 15 Pro Max 256GB, giá 34.990.000đ, màu titan..."
Node 2: "Chính sách bảo hành Apple 12 tháng, áp dụng cho..."
Node 3: "Samsung Galaxy S24 Ultra, RAM 12GB, pin 5000mAh..."
Code đọc file đó ra, tạo danh sách các nodes, rồi build BM25 index — hiểu đơn giản là tạo một bảng tra cứu từ khóa. Sau này khi khách hỏi "iPhone giá bao nhiêu", BM25 tra bảng đó và tìm ngay các nodes có chứa từ "iPhone", "giá".

if not os.path.exists(self.docstore_path):
    raise FileNotFoundError(...)
Nếu file không tồn tại → báo lỗi ngay, kèm hướng dẫn "chạy ingestion pipeline trước đi".

self._bm25_corpus = [node.text.split() for node in self._nodes]
Cắt mỗi đoạn văn thành danh sách từ. Ví dụ: "iPhone 15 Pro Max" → ["iPhone", "15", "Pro", "Max"]. BM25 cần dạng này để đếm từ.

if self._bm25_corpus:
    self._bm25 = BM25Okapi(self._bm25_corpus)
else:
    self._bm25 = BM25Okapi([[""]])
Nếu có dữ liệu → build index bình thường. Nếu không có gì → tạo index giả để code không crash.

Việc 2: Kết nối ChromaDB

def _init_vector(self) -> None:
ChromaDB là một kho lưu trữ vector — mỗi đoạn văn bản đã được chuyển thành một dãy số (vector) và lưu vào đây. Khi tìm kiếm, nó so sánh vector của câu hỏi với vector của các đoạn văn để tìm cái "gần nghĩa nhất".

existing = [c.name for c in db.list_collections()]
if self.collection_name not in existing:
    raise ValueError(...)
Kiểm tra xem kho dữ liệu có tồn tại không. Nếu không → báo lỗi rõ ràng thay vì crash khó hiểu.

Sau đó kết nối vào kho và tạo retriever sẵn sàng để dùng.

Phần 2 — Tìm kiếm: BM25 và Vector hoạt động thế nào
BM25 search — tìm theo từ khóa chính xác:

def _bm25_search(self, query: str) -> list[NodeWithScore]:
    tokenized_query = query.split()
    scores = self._bm25.get_scores(tokenized_query)
Query "IP15-PRM-256 giá bao nhiêu" → cắt thành ["IP15-PRM-256", "giá", "bao", "nhiêu"] → BM25 đếm xem mỗi node có bao nhiêu từ trùng → cho điểm → lấy top 20 node điểm cao nhất.

BM25 giỏi nhất khi khách gõ đúng từ khóa như mã SKU.

Vector search — tìm theo nghĩa:

def _vector_search(self, query: str) -> list[NodeWithScore]:
    return self._vector_retriever.retrieve(query)
Query được chuyển thành vector số, rồi tìm các node có vector gần nhất. Giỏi nhất khi khách hỏi kiểu mô tả như "điện thoại cao cấp của Apple" — dù không có từ "iPhone" nhưng vẫn tìm được.

Phần 3 — retrieve(): Gộp kết quả và trả về
Đây là hàm chính, làm 5 bước:

Bước 1 — Chạy song song

with ThreadPoolExecutor(max_workers=2) as executor:
    future_bm25 = executor.submit(self._bm25_search, query)
    future_vector = executor.submit(self._vector_search, query)
    bm25_results = future_bm25.result()
    vector_results = future_vector.result()
Thay vì tìm BM25 xong rồi mới tìm Vector (mất 2 giây), chạy cả hai cùng lúc (chỉ mất 1 giây). Giống như bạn vừa gọi điện vừa nhắn tin — làm song song nhanh hơn làm lần lượt.

Bước 2 — RRF Fusion: Gộp hai danh sách thành một

fusion_scores[node_id] += 1.0 / (self.rrf_k + rank)
Bây giờ có 2 danh sách kết quả, mỗi cái 20 nodes. Cần gộp lại thành 1 danh sách. Cách làm: mỗi node được cộng điểm dựa trên thứ hạng của nó trong từng danh sách.

Ví dụ thực tế:

Node "iPhone 15 Pro Max": hạng 3 trong BM25, hạng 2 trong Vector → điểm = 1/(60+3) + 1/(60+2) = 0.032
Node "Samsung S24": hạng 1 trong BM25, không có trong Vector → điểm = 1/(60+1) = 0.016
Node iPhone được cả hai retriever đồng thuận → điểm cao hơn → xếp trên. Đây là ý tưởng cốt lõi của RRF: node nào được nhiều nguồn đồng thuận thì đáng tin hơn.

Bước 3 — Sắp xếp, ưu tiên sản phẩm hơn chính sách

source_priority = {"product_catalog": 1, "policy_pdf": 0}
return (score, priority)
Nếu hai node có điểm bằng nhau, node từ catalog sản phẩm được xếp trên node từ file chính sách. Lý do: khách hỏi về sản phẩm thường muốn thông tin sản phẩm hơn.

Bước 4 — Cắt top 10

results = results[:top_k]
Lấy 10 kết quả tốt nhất, bỏ phần còn lại.

Bước 5 — SKU Boost: Đưa đúng sản phẩm lên đầu

query_tokens = query.split()
for i, nws in enumerate(results):
    product_code = nws.node.metadata.get("product_code", "")
    if product_code and product_code in query_tokens:
        boost_index = i
        break

if boost_index is not None and boost_index != 0:
    boosted = results.pop(boost_index)
    results.insert(0, boosted)
Nếu khách gõ đúng mã SKU như IP15-PRM-256, node của sản phẩm đó phải ở vị trí số 1 — dù điểm RRF của nó thấp hơn node khác.

Tại sao? Vì đây là business rule: khách gõ đúng mã → họ muốn đúng sản phẩm đó, không phải sản phẩm "gần giống".

Code tìm node có product_code trùng với một từ trong query, rồi kéo nó lên đầu danh sách.

Tóm lại toàn bộ luồng với ví dụ "IP15-PRM-256 giá bao nhiêu?":

1. BM25 tìm theo từ khóa → 20 nodes
2. Vector tìm theo nghĩa → 20 nodes     (chạy song song)
3. Gộp lại, tính điểm RRF → ~30 nodes unique
4. Sắp xếp theo điểm
5. Cắt còn 10
6. Thấy "IP15-PRM-256" trong query → kéo node đó lên đầu
7. Trả về 10 nodes

Giả sử khách gõ: "IP15-PRM-256 pin có tốt không"

Và sau bước RRF, danh sách results đang là:

[0] Node về "pin điện thoại nói chung"   ← điểm RRF cao nhất
[1] Node về "Samsung S24 pin 5000mAh"
[2] Node về "IP15-PRM-256 iPhone 15..."  ← đây mới là sản phẩm khách hỏi
[3] Node về "chính sách bảo hành"
Dòng 1

query_tokens = query.split()
Cắt câu hỏi thành danh sách từ: "IP15-PRM-256 pin có tốt không" → ["IP15-PRM-256", "pin", "có", "tốt", "không"]

Mục đích: để sau này kiểm tra xem mã SKU nào có trong câu hỏi không.

Dòng 2

boost_index: int | None = None
Tạo biến boost_index để ghi lại vị trí của node cần đưa lên đầu. Ban đầu là None — nghĩa là "chưa tìm thấy node nào cần boost".

Dòng 3–7

for i, nws in enumerate(results):
    product_code = nws.node.metadata.get("product_code", "")
    if product_code and product_code in query_tokens:
        boost_index = i
        break
Duyệt qua từng node trong danh sách, i là số thứ tự (0, 1, 2...), nws là node đó.

i=0: node "pin điện thoại" → product_code = "" → bỏ qua
i=1: node "Samsung S24" → product_code = "SAM-S24" → "SAM-S24" có trong ["IP15-PRM-256", "pin", ...] không? Không → bỏ qua
i=2: node "IP15-PRM-256" → product_code = "IP15-PRM-256" → "IP15-PRM-256" có trong ["IP15-PRM-256", "pin", ...] không? Có → boost_index = 2, dừng vòng lặp
Sau vòng lặp: boost_index = 2

Dòng 8–10

if boost_index is not None and boost_index != 0:
    boosted = results.pop(boost_index)
    results.insert(0, boosted)
Điều kiện: chỉ làm gì đó nếu tìm thấy node (is not None) VÀ node đó chưa ở đầu (!= 0). Nếu node đã ở vị trí 0 rồi thì không cần làm gì.

results.pop(2) — lấy node ở vị trí 2 ra khỏi danh sách, danh sách còn lại:

[0] Node về "pin điện thoại nói chung"
[1] Node về "Samsung S24 pin 5000mAh"
[2] Node về "chính sách bảo hành"
results.insert(0, boosted) — nhét node vừa lấy ra vào vị trí 0:

[0] Node về "IP15-PRM-256 iPhone 15..."  ← đã lên đầu
[1] Node về "pin điện thoại nói chung"
[2] Node về "Samsung S24 pin 5000mAh"
[3] Node về "chính sách bảo hành"
Dòng cuối

return results
Trả về danh sách đã được sắp xếp lại.

Tóm lại cả đoạn làm một việc: nếu khách gõ đúng mã SKU, đảm bảo sản phẩm đó luôn xuất hiện đầu tiên — dù thuật toán RRF xếp nó ở vị trí nào đi nữa.

---

### Điểm mấu chốt cần nhớ từ Batch 3

> `__init__` làm **tất cả công việc nặng** một lần duy nhất khi khởi tạo: đọc file, build index, kết nối DB. Sau đó `retrieve()` chỉ cần dùng các object đã sẵn sàng. Pattern này gọi là **eager initialization** — trả giá startup time để đổi lấy latency thấp khi query.

---

## Batch 4 — HybridRetriever: retrieve() — Trái tim của hệ thống

### Tổng quan 5 bước

```
retrieve(query, top_k=10)
    │
    ├── Bước 1: Parallel search (BM25 + Vector đồng thời)
    │
    ├── Bước 2: RRF Fusion (gộp kết quả, tính điểm)
    │
    ├── Bước 3: Sort + Tie-breaking (sắp xếp)
    │
    ├── Bước 4: top_k (cắt danh sách)
    │
    └── Bước 5: SKU Boost (đưa SKU khớp lên đầu)
```

---

### Bước 1: Parallel Search

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_bm25 = executor.submit(self._bm25_search, query)
    future_vector = executor.submit(self._vector_search, query)
    bm25_results: list[NodeWithScore] = future_bm25.result()
    vector_results: list[NodeWithScore] = future_vector.result()
```

**Tại sao parallel?** BM25 search và vector search hoàn toàn độc lập — không cần kết quả của cái này để chạy cái kia. Nếu BM25 mất 0.5s và vector search mất 1.5s:
- Tuần tự: 0.5 + 1.5 = **2.0s**
- Song song: max(0.5, 1.5) = **1.5s**

`ThreadPoolExecutor` dùng thread (không phải process) vì cả hai tác vụ đều là I/O-bound (đọc từ disk/memory), không phải CPU-bound. Python GIL không ảnh hưởng đến I/O-bound tasks.

`executor.submit()` trả về `Future` object ngay lập tức (non-blocking). `.result()` mới là blocking — chờ kết quả. Gọi cả hai `.submit()` trước, rồi mới gọi `.result()` — đây là cách đúng để chạy song song.

---

### Bước 2: RRF Fusion — Phần quan trọng nhất

```python
fusion_scores: dict[str, float] = {}
node_map: dict[str, NodeWithScore] = {}

for result_list in (bm25_results, vector_results):
    for rank, nws in enumerate(result_list, start=1):
        node_id = nws.node.node_id
        fusion_scores[node_id] = fusion_scores.get(node_id, 0.0) + 1.0 / (self.rrf_k + rank)
        if node_id not in node_map:
            node_map[node_id] = nws
```

**Công thức RRF:** `score += 1 / (k + rank)` với `k = 60`

**Tại sao k=60?** Đây là giá trị được chứng minh empirically trong paper gốc của RRF (Cormack et al., 2009). k=60 tạo ra "smooth curve" — sự khác biệt giữa rank 1 và rank 2 không quá lớn, tránh việc một retriever "thống trị" hoàn toàn.

**Ví dụ tính toán chi tiết:**

Giả sử có 3 nodes: A, B, C

| Node | BM25 rank | Vector rank | BM25 score | Vector score | Fusion score |
|---|---|---|---|---|---|
| A | 1 | 2 | 1/(60+1)=0.01639 | 1/(60+2)=0.01613 | **0.03252** |
| B | 2 | 1 | 1/(60+2)=0.01613 | 1/(60+1)=0.01639 | **0.03252** |
| C | 3 | - | 1/(60+3)=0.01587 | 0 | **0.01587** |

Node A và B có cùng fusion score (tie) → tie-breaking bằng `source_type`.
Node C chỉ xuất hiện trong BM25 → score thấp hơn nhiều.

**Tại sao không dùng raw score?** BM25 score và cosine similarity có scale hoàn toàn khác nhau (BM25 có thể là 15.3, cosine similarity là 0.87). Không thể cộng trực tiếp. RRF chỉ dùng **rank** (thứ hạng) nên không cần normalize — đây là ưu điểm lớn nhất của RRF.

**`node_map`** lưu NodeWithScore object gốc để sau này lấy metadata. Chỉ lưu lần đầu xuất hiện (`if node_id not in node_map`) — tránh overwrite với object từ retriever khác.

---

### Bước 3: Sort + Tie-breaking

```python
source_priority = {"product_catalog": 1, "policy_pdf": 0}

def sort_key(node_id: str):
    score = fusion_scores[node_id]
    priority = source_priority.get(
        node_map[node_id].node.metadata.get("source_type", ""), 0
    )
    return (score, priority)

sorted_ids = sorted(fusion_scores.keys(), key=sort_key, reverse=True)
```

`sort_key` trả về **tuple** `(score, priority)`. Python so sánh tuple theo thứ tự: so sánh phần tử đầu trước, nếu bằng nhau mới so sánh phần tử sau.

Ví dụ: Node A `(0.03252, 1)` vs Node B `(0.03252, 0)` → A > B vì `1 > 0` ở phần tử thứ hai.

**Business logic:** `product_catalog` (thông tin sản phẩm cụ thể) được ưu tiên hơn `policy_pdf` (chính sách chung) khi cùng điểm. Lý do: khách hỏi về sản phẩm thường muốn thông tin sản phẩm hơn là chính sách.

---

### Bước 4: top_k

```python
results = results[:top_k]
```

Đơn giản nhất trong 5 bước. Cắt list sau khi đã sort. `top_k` mặc định là 10 nhưng caller có thể truyền giá trị khác.

---

### Bước 5: SKU Boost — Business Rule quan trọng

```python
query_tokens = query.split()
boost_index: int | None = None
for i, nws in enumerate(results):
    product_code = nws.node.metadata.get("product_code", "")
    if product_code and product_code in query_tokens:
        boost_index = i
        break

if boost_index is not None and boost_index != 0:
    boosted = results.pop(boost_index)
    results.insert(0, boosted)
```

**Tại sao cần SKU Boost sau khi đã có RRF?**

RRF dựa trên rank từ BM25 và vector search. Nhưng đôi khi node có SKU khớp chính xác lại không được rank cao vì:
- BM25: node đó có nhiều từ khác không khớp query → score thấp
- Vector: embedding của node đó không gần embedding của query → score thấp

Ví dụ: Query `"IP15-PRM-256 pin có tốt không?"` — node về pin iPhone 15 Pro Max có thể rank cao hơn node về `IP15-PRM-256` vì từ "pin" xuất hiện nhiều lần. Nhưng business rule là: nếu khách gõ đúng mã SKU, node của SKU đó phải ở đầu.

**Logic chi tiết:**
- `query.split()` → `["IP15-PRM-256", "pin", "có", "tốt", "không?"]`
- Duyệt qua `results`, tìm node có `product_code` nằm trong list tokens
- `"IP15-PRM-256" in ["IP15-PRM-256", "pin", ...]` → True → `boost_index = i`
- `results.pop(i)` lấy node ra, `results.insert(0, boosted)` đưa lên đầu

**`boost_index != 0`** — không cần boost nếu node đã ở vị trí 0 rồi.

---

### Điểm mấu chốt cần nhớ từ Batch 4

> `retrieve()` là sự kết hợp của **kỹ thuật** (parallel execution, RRF) và **business logic** (tie-breaking, SKU boost). Hai loại này được tách biệt rõ ràng trong code: RRF là thuật toán chung, SKU boost là rule đặc thù của domain bán hàng. Hiểu sự phân tách này giúp bạn biết chỗ nào cần thay đổi khi business thay đổi.

---

## Batch 5 — RAGPipeline: Điều phối và Routing

### Vị trí trong pipeline

```
User Query → RelevanceChecker → HybridRetriever → Kết quả
                    ↑_________________________↑
              [RAGPipeline điều phối cả hai]
                    ↑
              Batch 5 phân tích phần này
```

RAGPipeline là **orchestrator** — nó không làm gì nặng, chỉ gọi hai thành phần kia theo đúng thứ tự và routing kết quả.

---

### Toàn bộ class

```python
class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, checker: RelevanceChecker) -> None:
        self.retriever = retriever
        self.checker = checker

    def query(self, user_query: str) -> Union[str, list[NodeWithScore]]:
        label = self.checker.check(user_query)
        logger.info(f"[RelevanceChecker] label={label} query={user_query!r}")

        if label == "NO_MATCH":
            return _DEFAULT_NO_MATCH_RESPONSE

        return self.retriever.retrieve(user_query)
```

Class này chỉ có ~10 dòng code thực sự. Đây là **intentional simplicity** — RAGPipeline không nên biết chi tiết về BM25 hay LLM. Nó chỉ biết: "hỏi checker, nếu không liên quan thì trả lời mặc định, nếu liên quan thì hỏi retriever".

---

### Return type: `Union[str, list[NodeWithScore]]`

Hàm `query()` có thể trả về **hai loại khác nhau**:
- `str` — khi `NO_MATCH`, trả về chuỗi thông báo
- `list[NodeWithScore]` — khi `CAN_ANSWER`/`PARTIAL`, trả về danh sách nodes

Đây là **polymorphic return type**. Caller phải kiểm tra kiểu trước khi dùng:

```python
result = pipeline.query("iPhone 15 giá bao nhiêu?")
if isinstance(result, str):
    print(result)  # "Xin lỗi, câu hỏi nằm ngoài phạm vi..."
else:
    for node in result:
        print(node.node.text)  # nội dung tài liệu
```

---

### Logging

```python
logger.info(f"[RelevanceChecker] label={label} query={user_query!r}")
```

`{user_query!r}` dùng `repr()` — hiển thị chuỗi với dấu nháy và escape characters. Ví dụ: query `"iPhone\n15"` sẽ log là `'iPhone\n15'` thay vì xuống dòng thật. Điều này giúp log dễ đọc và debug hơn.

Log này quan trọng trong production để:
- Theo dõi phân phối labels (bao nhiêu % là NO_MATCH?)
- Debug khi checker phân loại sai
- Audit trail cho câu hỏi của khách hàng

---

### Default response

```python
_DEFAULT_NO_MATCH_RESPONSE = (
    "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của hệ thống. "
    "Vui lòng hỏi về sản phẩm hoặc chính sách bảo hành/đổi trả."
)
```

Được định nghĩa ở **module level** (ngoài class), không phải trong `__init__`. Lý do: đây là hằng số, không thay đổi theo instance. Đặt ở module level giúp dễ tìm và thay đổi khi cần.

---

### Điểm mấu chốt cần nhớ từ Batch 5

> RAGPipeline minh họa nguyên tắc **Single Responsibility**: mỗi class chỉ làm một việc. RelevanceChecker phân loại, HybridRetriever tìm kiếm, RAGPipeline điều phối. Khi cần thay đổi logic tìm kiếm, bạn chỉ sửa HybridRetriever mà không cần đụng vào RAGPipeline hay RelevanceChecker.
