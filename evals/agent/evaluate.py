"""Repeatable multi-turn quality gate for the shopping advisor.

The suite expands 10 behavioral templates into 200 deterministic cases. It is
designed for CI: factual/state assertions are code-based, while prose quality
can be judged separately without weakening the hard safety gates.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.api.main import app


@dataclass
class Metric:
    passed: int = 0
    total: int = 0

    def record(self, result: bool) -> None:
        self.total += 1
        self.passed += int(result)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


TEMPLATES = (
    ("Nên chọn Dell hay Asus tầm 20 triệu?", "comparison"),
    ("So sánh Dell với MacBook cùng tầm giá 20 triệu", "comparison"),
    ("Tư vấn laptop gần 20 triệu", "catalog_search"),
    ("Máy nào có hiệu năng trên giá tốt nhất tầm 20 triệu?", "value_ranking"),
    ("Hãy tư vấn chi tiết sản phẩm mã 00928595.", "product_detail"),
    ("Tại sao Dell lại rẻ hơn MacBook với cùng hiệu năng?", "price_causality"),
    ("Có laptop nào tầm 20 triệu có card rời không?", "catalog_search"),
    ("Tư vấn điện thoại tầm 5 triệu", "catalog_search"),
    ("Apple bảo hành bao lâu?", "policy"),
    ("Có mẫu nào rẻ hơn 00928595 không?", "cheaper_alternatives"),
    ("Tư vấn máy cho tôi", "clarify"),
)


def run() -> dict:
    client = TestClient(app)
    metrics = {
        "intent": Metric(),
        "category": Metric(),
        "budget": Metric(),
        "grounded_products": Metric(),
        "latency_under_6s": Metric(),
        "context_continuity": Metric(),
        "candidate_preservation": Metric(),
        "decision_quality": Metric(),
        "packet_verification": Metric(),
        "topic_isolation": Metric(),
    }
    latencies: list[float] = []
    request_count = 0

    for index in range(220):
        message, expected_intent = TEMPLATES[index % len(TEMPLATES)]
        started = time.perf_counter()
        response = client.post(
            "/api/chat",
            json={"message": message, "history": [], "conversation_state": None},
        )
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        payload = response.json()
        request_count += 1

        metrics["intent"].record(payload["answer_type"] == expected_intent)
        expected_category = (
            "Mobile Phone" if "điện thoại" in message.lower() else None
        )
        metrics["category"].record(
            expected_category is None
            or all(
                product["category"] == expected_category
                for product in payload["products"]
            )
        )
        metrics["budget"].record(
            "20 triệu" not in message
            or payload["active_context"]["budget_target"] == 20_000_000
        )
        returned_codes = {
            product["code"] for product in payload["products"]
        }
        source_codes = {
            source.get("product_code")
            for source in payload["sources"]
            if source.get("product_code")
        }
        metrics["grounded_products"].record(returned_codes <= source_codes)
        metrics["latency_under_6s"].record(elapsed < 6.0)
        if expected_intent in {"comparison", "value_ranking", "price_causality"}:
            normalized = payload["text"].lower()
            metrics["decision_quality"].record(
                any(term in normalized for term in ("kết luận", "lý do", "vì sao"))
            )
        if expected_intent in {"comparison", "price_causality"}:
            metrics["packet_verification"].record(
                bool(payload["verification"])
                and payload["verification"]["approved"] is True
                and "decision_packet_verifier" in payload["tools_used"]
            )

    follow_ups = (
        "Ưu tiên độ bền và hiệu năng.",
        "Máy nào đáng tiền hơn?",
        "Tại sao Dell lại rẻ hơn?",
        "Nếu chơi game thì sao?",
    )
    for _ in range(20):
        first = client.post(
            "/api/chat",
            json={
                "message": "Nên chọn Dell hay Asus tầm 20 triệu?",
                "history": [],
                "conversation_state": None,
            },
        ).json()
        request_count += 1
        state = first["conversation_state"]
        expected_codes = set(first["active_context"]["candidate_codes"])
        for follow_up in follow_ups:
            payload = client.post(
                "/api/chat",
                json={
                    "message": follow_up,
                    "history": [],
                    "conversation_state": state,
                },
            ).json()
            request_count += 1
            metrics["context_continuity"].record(
                payload["active_context"]["budget_target"] == 20_000_000
                and payload["active_context"]["compared_brands"] == ["Dell", "Asus"]
            )
            metrics["candidate_preservation"].record(
                set(payload["active_context"]["candidate_codes"]) == expected_codes
            )
            state = payload["conversation_state"]
        detail = client.post(
            "/api/chat",
            json={
                "message": "Hãy tư vấn chi tiết sản phẩm mã 00928595.",
                "history": [],
                "conversation_state": state,
            },
        ).json()
        request_count += 1
        metrics["topic_isolation"].record(
            detail["active_context"]["candidate_codes"] == ["00928595"]
            and detail["active_context"]["compared_brands"] == []
            and detail["active_context"]["preferences"] == {}
        )

    sorted_latency = sorted(latencies)
    report = {
        "cases": request_count,
        "metrics": {
            name: {
                "passed": metric.passed,
                "total": metric.total,
                "rate": round(metric.rate, 4),
            }
            for name, metric in metrics.items()
        },
        "p95_seconds": round(sorted_latency[int(len(sorted_latency) * 0.95) - 1], 4),
    }
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    hard_failures = [
        name
        for name in ("category", "budget", "grounded_products")
        if report["metrics"][name]["rate"] < 1.0
    ]
    if report["metrics"]["intent"]["rate"] < 0.95:
        hard_failures.append("intent")
    if report["metrics"]["context_continuity"]["rate"] < 0.98:
        hard_failures.append("context_continuity")
    if report["metrics"]["candidate_preservation"]["rate"] < 0.98:
        hard_failures.append("candidate_preservation")
    if report["metrics"]["decision_quality"]["rate"] < 0.90:
        hard_failures.append("decision_quality")
    if report["metrics"]["packet_verification"]["rate"] < 1.0:
        hard_failures.append("packet_verification")
    if report["metrics"]["topic_isolation"]["rate"] < 1.0:
        hard_failures.append("topic_isolation")
    if report["metrics"]["latency_under_6s"]["rate"] < 0.95:
        hard_failures.append("latency_under_6s")
    raise SystemExit(1 if hard_failures else 0)
