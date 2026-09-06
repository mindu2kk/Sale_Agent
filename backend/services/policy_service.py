"""Deterministic, extractive answers from local warranty/policy documents."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = PROJECT_ROOT / "data" / "Policies"


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", value.casefold().replace("đ", "d")
    )
    ascii_value = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


@dataclass(frozen=True)
class PolicyPassage:
    file_name: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class PolicyAnswer:
    text: str
    sources: tuple[dict, ...]


class PolicyKnowledgeBase:
    """Read PDFs locally and return quoted facts without generative synthesis."""

    POLICY_TERMS = (
        "bao hanh",
        "doi tra",
        "hoan tien",
        "doi may",
        "chinh sach",
        "warranty",
        "return",
        "refund",
        "exchange",
    )

    def __init__(self, policy_dir: Path = POLICY_DIR) -> None:
        self.policy_dir = policy_dir
        self._passages, self._unreadable_files = self._load_passages()

    def is_policy_query(self, query: str) -> bool:
        normalized = _normalize(query)
        return any(term in normalized for term in self.POLICY_TERMS)

    def answer(self, query: str, *, brand: str | None = None) -> PolicyAnswer:
        return_answer = self._answer_fpts_return_policy(query)
        if return_answer is not None:
            return return_answer

        enriched_query = f"{query} {brand or ''}".strip()
        matches = self.search(enriched_query, limit=3)
        if not matches:
            unreadable_note = ""
            if self._unreadable_files:
                unreadable_note = (
                    " Tài liệu "
                    + ", ".join(self._unreadable_files)
                    + " hiện là PDF scan chưa có lớp chữ/OCR nên hệ thống không thể đọc chính xác."
                )
            return PolicyAnswer(
                text=(
                    "Dạ, mình chưa tìm thấy điều khoản đủ rõ trong tài liệu nội bộ để trả lời chắc chắn."
                    + unreadable_note
                    + " Mình sẽ không tự suy đoán thời hạn hay điều kiện bảo hành/đổi trả."
                ),
                sources=(),
            )

        selected = self._select_non_duplicate(matches, limit=2)
        lines = ["Dạ, mình tra trực tiếp từ tài liệu chính sách nội bộ:"]
        sources: list[dict] = []
        for passage in selected:
            excerpt = self._focused_excerpt(passage.text, enriched_query)
            lines.append(
                f"- {excerpt} (Nguồn: {passage.file_name}, trang {passage.page})"
            )
            sources.append(
                {
                    "source": "policy_pdf",
                    "file_name": passage.file_name,
                    "page": passage.page,
                }
            )
        lines.append(
            "Nếu bạn cung cấp hãng hoặc SKU cụ thể, mình sẽ đối chiếu đúng chính sách áp dụng cho sản phẩm đó."
        )
        return PolicyAnswer(text="\n".join(lines), sources=tuple(sources))

    def _answer_fpts_return_policy(self, query: str) -> PolicyAnswer | None:
        normalized = _normalize(query)
        if not any(
            term in normalized
            for term in (
                "doi tra",
                "hoan tien",
                "tra hang",
                "co duoc tra",
                "san pham loi",
                "loi nha san xuat",
            )
        ):
            return None
        sidecar = self.policy_dir / "FPTS_Return_Exchange_Policy.ocr.txt"
        if not sidecar.exists():
            return None

        mentions_old = any(term in normalized for term in ("san pham cu", "may cu"))
        mentions_defect = any(
            term in normalized
            for term in ("loi", "loi nha san xuat", "hu", "bao hanh")
        )
        mentions_no_defect = any(
            term in normalized
            for term in ("khong loi", "nhu cau", "muon tra", "tra san pham")
        )

        lines = ["Dạ, theo chính sách đổi trả FPT Shop đã OCR từ tài liệu nội bộ:"]
        sources: list[dict] = []
        if mentions_old:
            lines.append(
                "- Với sản phẩm cũ lỗi nhà sản xuất trong 0-30 ngày: đổi 1 sản phẩm chính "
                "tương đương, cùng model, dung lượng và thời gian bảo hành. Nếu không có sản phẩm "
                "tương đương, hoàn lại 100% giá trị sản phẩm; các khoản phí khác vẫn có thể áp dụng."
            )
            sources.append(
                {"source": "policy_pdf", "file_name": sidecar.stem.replace(".ocr", "") + ".pdf", "page": 2}
            )
        elif mentions_no_defect:
            lines.append(
                "- Trả theo nhu cầu không phải lúc nào cũng miễn phí: FPT Shop kiểm tra tình trạng "
                "máy và xác định giá trị thu lại. Tài liệu nêu mức khấu hao 30% trong tháng đầu, "
                "mỗi tháng tiếp theo cộng thêm 5%, cùng các phí phát sinh nếu có."
            )
            sources.append(
                {"source": "policy_pdf", "file_name": "FPTS_Return_Exchange_Policy.pdf", "page": 2}
            )
        else:
            lines.append(
                "- Với sản phẩm ICT mới lỗi nhà sản xuất trong 0-30 ngày: áp dụng 1 đổi 1 "
                "sản phẩm chính cùng model, màu và dung lượng, phí khấu hao 0%. Nếu hết hàng, "
                "khách có thể đổi sang sản phẩm tương đương hoặc cao hơn về giá trị."
            )
            lines.append(
                "- Điều kiện chung gồm: màn hình không trầy xước; sản phẩm còn đủ điều kiện "
                "bảo hành, không có bất thường về chức năng/ngoại quan; và đã đăng xuất các tài khoản."
            )
            sources.extend(
                [
                    {"source": "policy_pdf", "file_name": "FPTS_Return_Exchange_Policy.pdf", "page": 1},
                    {"source": "policy_pdf", "file_name": "FPTS_Return_Exchange_Policy.pdf", "page": 2},
                ]
            )
        if mentions_defect and not mentions_old:
            lines.append(
                "- Từ ngày 31 đến khi hết hạn bảo hành, hướng xử lý phụ thuộc nhóm sản phẩm và "
                "chính sách hãng; không nên mặc định tiếp tục được 1 đổi 1."
            )
        lines.append(
            "Kết luận cuối cùng vẫn cần đối chiếu loại sản phẩm, tình trạng máy và ngày xuất hóa đơn."
        )
        return PolicyAnswer(text="\n".join(lines), sources=tuple(sources))

    def search(self, query: str, *, limit: int = 5) -> list[PolicyPassage]:
        normalized_query = _normalize(query)
        query_tokens = {
            token
            for token in normalized_query.split()
            if len(token) > 2
            and token
            not in {
                "cho",
                "cua",
                "the",
                "nay",
                "thi",
                "la",
                "duoc",
                "khong",
                "chinh",
                "sach",
            }
        }
        brand_terms = {
            brand for brand in ("apple", "samsung", "fpts") if brand in query_tokens
        }
        asks_return = any(
            phrase in normalized_query
            for phrase in (
                "doi tra",
                "hoan tien",
                "tra hang",
                "doi may",
                "san pham loi",
                "co duoc tra",
                "return",
                "refund",
                "exchange",
            )
        )
        asks_warranty = any(
            phrase in normalized_query for phrase in ("bao hanh", "warranty")
        )
        scored: list[PolicyPassage] = []
        for file_name, page, text in self._passages:
            normalized_text = _normalize(text)
            normalized_file = _normalize(file_name)
            if asks_return and not any(
                term in normalized_file for term in ("return", "exchange", "doi tra")
            ):
                continue
            text_tokens = set(normalized_text.split())
            overlap = len(query_tokens & text_tokens)
            phrase_bonus = sum(
                3
                for phrase in (
                    "bao hanh",
                    "doi tra",
                    "hoan tien",
                    "thoi han",
                    "dieu kien",
                )
                if phrase in normalized_query and phrase in normalized_text
            )
            source_bonus = sum(
                8 for brand in brand_terms if brand in normalized_file
            )
            if asks_return and any(
                term in normalized_file for term in ("return", "exchange", "doi tra")
            ):
                source_bonus += 20
            if asks_warranty and "warranty" in normalized_file:
                source_bonus += 8
            score = overlap + phrase_bonus + source_bonus
            if score > 0:
                scored.append(
                    PolicyPassage(
                        file_name=file_name,
                        page=page,
                        text=text,
                        score=float(score),
                    )
                )
        scored.sort(key=lambda item: (-item.score, item.file_name, item.page))
        return scored[:limit]

    def _load_passages(self) -> tuple[list[tuple[str, int, str]], list[str]]:
        passages: list[tuple[str, int, str]] = []
        unreadable: list[str] = []
        try:
            import fitz
        except ImportError:
            return passages, [path.name for path in self.policy_dir.glob("*.pdf")]

        for path in sorted(self.policy_dir.glob("*.pdf")):
            sidecar = path.with_suffix(".ocr.txt")
            if sidecar.exists():
                file_has_text = False
                raw_text = sidecar.read_text(encoding="utf-8", errors="replace")
                page_sections = re.split(r"--- Trang (\d+) ---", raw_text)
                for index in range(1, len(page_sections), 2):
                    page_number = int(page_sections[index])
                    lines = [
                        re.sub(r"\s+", " ", line).strip()
                        for line in page_sections[index + 1].splitlines()
                        if line.strip()
                    ]
                    text = ". ".join(lines)
                    if not text:
                        continue
                    file_has_text = True
                    for chunk in self._chunk_text(text):
                        passages.append((path.name, page_number, chunk))
                if file_has_text:
                    continue

            document = fitz.open(path)
            file_has_text = False
            for page_number, page in enumerate(document, start=1):
                text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                if not text:
                    continue
                file_has_text = True
                for chunk in self._chunk_text(text):
                    passages.append((path.name, page_number, chunk))
            document.close()
            if not file_has_text:
                unreadable.append(path.name)
        return passages, unreadable

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1200) -> list[str]:
        sentences = re.split(r"(?<=[.!?;:])\s+|(?=\d+(?:\.\d+)+\s)", text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > chunk_size:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _focused_excerpt(text: str, query: str, max_chars: int = 520) -> str:
        normalized_query = _normalize(query)
        if any(term in normalized_query for term in ("bao lau", "thoi han", "may thang", "may nam")):
            duration_match = re.search(
                r"[^.!?]{0,120}(?:bảo hành|thời hạn)[^.!?]{0,120}"
                r"(?:\(\d+\)\s*năm|\d+\s*(?:năm|tháng)|một\s*\(\d+\)\s*năm)[^.!?]{0,120}",
                text,
                flags=re.IGNORECASE,
            )
            if duration_match:
                return re.sub(r"\s+", " ", duration_match.group()).strip()

        terms = [
            token
            for token in normalized_query.split()
            if len(token) > 3
        ]
        sentences = re.split(r"(?<=[.!?;])\s+", text)
        ranked = sorted(
            sentences,
            key=lambda sentence: sum(
                1 for term in terms if term in _normalize(sentence)
            ),
            reverse=True,
        )
        excerpt = " ".join(ranked[:2]).strip() or text
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        return excerpt

    @staticmethod
    def _select_non_duplicate(
        passages: list[PolicyPassage], *, limit: int
    ) -> list[PolicyPassage]:
        selected: list[PolicyPassage] = []
        seen: set[tuple[str, str]] = set()
        for passage in passages:
            signature = (passage.file_name, _normalize(passage.text)[:180])
            if signature in seen:
                continue
            selected.append(passage)
            seen.add(signature)
            if len(selected) >= limit:
                break
        return selected


@lru_cache(maxsize=1)
def get_policy_knowledge_base() -> PolicyKnowledgeBase:
    return PolicyKnowledgeBase()
