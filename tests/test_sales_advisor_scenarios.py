import pytest
from backend.services.catalog import CatalogProduct
from backend.harness.types import ConversationPlan, BeliefState, EvidenceRef
from backend.harness.fallback import build_verified_fallback_response


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_candidates():
    """Three laptops with *no* durability/warranty evidence fields."""
    return [
        CatalogProduct(
            code="LAP001", context="Dell XPS 13", brand="Dell",
            price="25000000", category="laptop",
            specs=("Core i7", "16GB RAM", "512GB SSD"),
            title="Dell XPS 13",
        ),
        CatalogProduct(
            code="LAP002", context="MacBook Air M2", brand="Apple",
            price="26000000", category="laptop",
            specs=("M2", "8GB RAM", "256GB SSD"),
            title="MacBook Air M2",
        ),
        CatalogProduct(
            code="LAP003", context="Asus ROG Zephyrus", brand="Asus",
            price="35000000", category="laptop",
            specs=("Ryzen 9", "32GB RAM", "RTX 3070"),
            title="Asus ROG Zephyrus",
        ),
    ]


@pytest.fixture
def mixed_perf_candidates():
    """One strong GPU candidate, one basic candidate — performance must NOT
    blanket-claim all are strong."""
    return [
        CatalogProduct(
            code="G001", context="Gaming", brand="MSI",
            price="30000000", category="laptop",
            specs=("Core i7", "16GB RAM", "RTX 4060"),
            title="MSI Gaming GF63",
        ),
        CatalogProduct(
            code="B001", context="Basic", brand="HP",
            price="10000000", category="laptop",
            specs=("Celeron N4020", "4GB RAM", "128GB SSD"),
            title="HP 14-Basic",
        ),
    ]


@pytest.fixture
def weak_spec_candidates():
    """Laptops with only basic specs — no GPU rời, no strong CPU."""
    return [
        CatalogProduct(
            code="W001", context="Basic", brand="HP",
            price="10000000", category="laptop",
            specs=("Celeron N4020", "4GB RAM", "128GB SSD"),
            title="HP 14-Basic",
        ),
        CatalogProduct(
            code="W002", context="Basic", brand="Lenovo",
            price="11000000", category="laptop",
            specs=("Pentium Silver", "4GB RAM", "256GB SSD"),
            title="Lenovo IdeaPad 1",
        ),
    ]


@pytest.fixture
def real_code_candidate():
    """A candidate with a realistic product code like 00927195."""
    return CatalogProduct(
        code="00927195", context="MacBook Pro 14", brand="Apple",
        price="55000000", category="laptop",
        specs=("M5 Pro", "18GB RAM", "512GB SSD"),
        title="MacBook Pro 14 M5 2026",
    )


@pytest.fixture
def base_plan():
    return ConversationPlan(
        intent="recommend_by_need",
        skillName="sales_advisor",
        objective="Find suitable laptop",
        shouldAskClarification=False,
    )


@pytest.fixture
def base_context():
    return BeliefState(
        version=1,
        confidence=0.9,
        freshness="fresh",
        catalogRevision="rev1",
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fb(candidates, evidence, context, plan, user_query,
        mode="consultative", reason="test_failure"):
    return build_verified_fallback_response(
        reason=reason,
        candidates=candidates,
        evidence_refs=evidence,
        plan=plan,
        context=context,
        mode=mode,
        user_query=user_query,
        ai_available=mode != "catalog_fallback",
    )


def _assert_no(msg, forbidden):
    for f in forbidden:
        assert f not in msg, f"Forbidden phrase '{f}' found in:\n  {msg}"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Product detail — exact code matching
# ═══════════════════════════════════════════════════════════════════════════

class TestProductDetail:
    def test_exact_match_LAP001(self, base_candidates, base_plan, base_context):
        msg = _fb([base_candidates[0]], [], base_context, base_plan,
                   "Hãy tư vấn chi tiết sản phẩm mã LAP001")
        assert "Dell XPS 13" in msg
        assert "LAP001" in msg
        assert "Core i7" in msg
        _assert_no(msg, ["trong tầm ngân sách này", "do thông tin về giá"])

    def test_exact_match_LAP002(self, base_candidates, base_plan, base_context):
        msg = _fb([base_candidates[1]], [], base_context, base_plan,
                   "Hãy tư vấn chi tiết sản phẩm mã LAP002")
        assert "MacBook Air M2" in msg
        assert "LAP002" in msg
        assert "Dell XPS 13" not in msg

    def test_real_code_00927195(self, real_code_candidate, base_candidates,
                                base_plan, base_context):
        """Query for 00927195 with that candidate in the list must return
        MacBook Pro 14 M5, NOT Dell XPS."""
        all_cands = base_candidates + [real_code_candidate]
        msg = _fb(all_cands, [], base_context, base_plan,
                   "Hãy tư vấn chi tiết sản phẩm mã 00927195")
        assert "MacBook Pro 14 M5 2026" in msg
        assert "00927195" in msg
        assert "Dell XPS 13" not in msg

    def test_code_mismatch_picks_none(self, base_candidates, base_plan,
                                      base_context):
        """When requested code doesn't match any candidate, must say not found
        instead of picking candidates[0]."""
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Hãy tư vấn chi tiết sản phẩm mã ZZZZZ999")
        assert "chưa tìm thấy" in msg.lower()
        assert "ZZZZZ999" in msg
        assert "Dell XPS" not in msg
        assert "MacBook" not in msg
        assert "Asus" not in msg

    def test_unknown_code_no_candidates(self, base_plan, base_context):
        msg = _fb([], [], base_context, base_plan,
                   "Hãy tư vấn chi tiết sản phẩm mã UNKNOWN99")
        assert "chưa tìm thấy" in msg.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Durability — evidence-gated claims
# ═══════════════════════════════════════════════════════════════════════════

class TestDurability:
    def test_no_evidence_no_claim(self, base_candidates, base_plan, base_context):
        """Without durability/warranty evidence, must NOT claim durability
        or 'dễ bảo hành'."""
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Tôi cần laptop bền bỉ")
        _assert_no(msg, [
            "độ bền được đánh giá khá cao",
            "độ bền cao",
            "được đánh giá cao",
            "dễ bảo hành",
        ])
        # "bền nhất" OK only in negation
        if "bền nhất" in msg:
            assert "chưa" in msg.split("bền nhất")[0][-30:]
        assert "chưa thấy dữ liệu độ bền" in msg or "chưa chốt" in msg
        # Must suggest checking warranty externally
        assert "kiểm tra thêm chính sách bảo hành" in msg
        # Follow-up
        assert "mang đi học/công tác" in msg

    def test_with_durability_evidence(self, base_candidates, base_plan, base_context):
        dur_ev = [EvidenceRef(
            evidenceId="dur1", source="catalog", field="durability",
            value="MIL-STD-810H", fetchedAt="2026-01-01",
            catalogRevision="rev1", trust="high", freshness="fresh",
        )]
        msg = _fb(base_candidates, dur_ev, base_context, base_plan,
                   "Tôi cần laptop bền bỉ")
        assert "vật liệu" in msg or "chứng nhận" in msg or "độ bền" in msg

    def test_with_warranty_evidence(self, base_candidates, base_plan, base_context):
        """With warranty evidence, CAN mention warranty."""
        war_ev = [EvidenceRef(
            evidenceId="w1", source="catalog", field="warranty",
            value="3 năm", fetchedAt="2026-01-01",
            catalogRevision="rev1", trust="high", freshness="fresh",
        )]
        msg = _fb(base_candidates, war_ev, base_context, base_plan,
                   "Tôi cần laptop bền bỉ")
        assert "bảo hành" in msg

    def test_stale_price_no_leak(self, base_candidates, base_plan, base_context):
        ev = [EvidenceRef(
            evidenceId="p1", source="catalog", field="price",
            value=25000000, fetchedAt="2020",
            catalogRevision="old", trust="medium", freshness="stale",
        )]
        msg = _fb(base_candidates, ev, base_context, base_plan,
                   "Tôi cần laptop bền bỉ", reason="stale_evidence")
        _assert_no(msg, [
            "do thông tin về giá hoặc khuyến mãi của hệ thống đang cập nhật",
            "trong tầm ngân sách này",
        ])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Performance — per-candidate highlights
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    def test_strong_specs_per_candidate(self, base_candidates, base_plan, base_context):
        """ROG Zephyrus (RTX 3070) should be highlighted specifically."""
        msg = _fb(base_candidates, [], base_context, base_plan, "Cấu hình mạnh")
        # Must contain per-candidate info, not blanket "tất cả đều mạnh"
        assert "Asus ROG Zephyrus" in msg
        assert "GPU rời" in msg
        assert "chơi game, dựng video" in msg
        _assert_no(msg, ["cấu hình rất mạnh mẽ"])

    def test_mixed_candidates_no_blanket(self, mixed_perf_candidates, base_plan,
                                         base_context):
        """Mixed list: MSI (RTX 4060) + HP Basic (Celeron).
        Must NOT imply both are strong."""
        msg = _fb(mixed_perf_candidates, [], base_context, base_plan,
                   "Cấu hình mạnh")
        assert "MSI Gaming GF63" in msg
        assert "HP 14-Basic" in msg
        # HP Basic must be flagged as weak, not strong
        assert "chưa rõ" in msg or "chưa thấy" in msg

    def test_weak_specs_no_strong_claim(self, weak_spec_candidates, base_plan,
                                        base_context):
        msg = _fb(weak_spec_candidates, [], base_context, base_plan,
                   "Cấu hình mạnh")
        _assert_no(msg, ["rất mạnh mẽ"])
        assert "chưa thấy" in msg


# ═══════════════════════════════════════════════════════════════════════════
# 4. Existing scenarios (budget, comparison, hardware, superlative, etc.)
# ═══════════════════════════════════════════════════════════════════════════

class TestBudget:
    def test_budget_consulting(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Tư vấn laptop gần 20 triệu")
        assert "trong tầm ngân sách này" in msg
        assert "rất đáng cân nhắc" in msg

    def test_stale_price_budget(self, base_candidates, base_plan, base_context):
        ev = [EvidenceRef(
            evidenceId="1", source="catalog", field="price",
            value=25000000, fetchedAt="2020",
            catalogRevision="old", trust="medium", freshness="stale",
        )]
        msg = _fb(base_candidates, ev, base_context, base_plan,
                   "Tư vấn laptop gần 20 triệu", reason="stale_evidence")
        assert "Giá có thể cần kiểm tra lại ở thời điểm mua" in msg


class TestComparison:
    def test_two(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates[:2], [], base_context, base_plan,
                   "So sánh A và B")
        assert "đều có ưu điểm riêng" in msg
        assert "giá, hiệu năng, pin" in msg


class TestHardware:
    def test_explanation(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "RTX 3050 dùng để làm gì?")
        assert "đang sử dụng linh kiện này" in msg


class TestSuperlative:
    def test_no_candidates(self, base_plan, base_context):
        msg = _fb([], [], base_context, base_plan, "Con nào bền nhất?")
        assert "chưa tìm thấy" in msg

    def test_with_candidates(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Con nào mạnh nhất?")
        assert "chưa đủ dữ liệu" in msg or "mỗi mẫu đều có thế mạnh" in msg


class TestAIUnavailable:
    def test_with_candidates(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Tư vấn laptop", mode="catalog_fallback")
        assert "Hệ thống AI phân tích chuyên sâu đang bận một chút" in msg

    def test_no_candidates(self, base_plan, base_context):
        msg = _fb([], [], base_context, base_plan,
                   "Tư vấn laptop", mode="catalog_fallback")
        assert "Mình chưa thể tìm thấy thông tin chính xác" in msg
        _assert_no(msg, ["Hệ thống AI phân tích"])


class TestPhone:
    def test_phone(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Gợi ý điện thoại")
        assert "mình có các mẫu điện thoại" in msg


# ═══════════════════════════════════════════════════════════════════════════
# 5. Negative phrase guards (cross-intent)
# ═══════════════════════════════════════════════════════════════════════════

class TestNegativePhraseGuards:
    def test_durability_no_price(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Tôi cần laptop bền bỉ")
        _assert_no(msg, [
            "trong tầm ngân sách này",
            "do thông tin về giá",
            "mỏng nhẹ hay cấu hình mạnh",
        ])

    def test_performance_no_price(self, base_candidates, base_plan, base_context):
        msg = _fb(base_candidates, [], base_context, base_plan,
                   "Cấu hình mạnh")
        _assert_no(msg, ["trong tầm ngân sách này", "do thông tin về giá"])

    def test_product_no_budget(self, base_candidates, base_plan, base_context):
        msg = _fb([base_candidates[0]], [], base_context, base_plan,
                   "Chi tiết sản phẩm mã LAP001")
        _assert_no(msg, ["trong tầm ngân sách này", "do thông tin về giá"])
