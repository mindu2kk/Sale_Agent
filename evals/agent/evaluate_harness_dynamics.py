"""Dynamic trajectory gates for the harness control plane."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.main import app


SCENARIOS = (
    ("Tư vấn laptop tầm 20 triệu", "catalog-search"),
    ("So sánh Oppo A6C 4GB, Tecno Spark 50 4GB", "product-comparison"),
    ("Hãy tư vấn chi tiết sản phẩm mã 00928595", "product-detail"),
    ("Có máy nào hiệu năng trên giá tốt nhất?", "value-ranking"),
    ("Tôi cần laptop xịn nhất shop của bạn", "catalog-ranking"),
)


def main() -> int:
    os.environ["EXPOSE_DECISION_TRACE"] = "true"
    client = TestClient(app)
    passed = 0
    failures: list[str] = []
    for query, expected_skill in SCENARIOS:
        response = client.post(
            "/api/chat",
            json={
                "message": query,
                "history": [],
                "conversation_state": None,
            },
            headers={"x-eval-mode": "harness"},
        )
        payload = response.json()
        trace = (payload.get("decision_trace") or {}).get("harness")
        if response.status_code != 200:
            failures.append(f"{query}: HTTP {response.status_code}")
            continue
        if trace is None:
            # Run with EXPOSE_DECISION_TRACE=true for trajectory evaluation.
            failures.append(f"{query}: missing harness trace")
            continue
        checks = (
            trace["skill"]["name"] == expected_skill,
            trace["events"][-1]["phase"] == "commit",
            not any(
                item["severity"] == "critical"
                for item in trace["governance"].get("postflight", [])
            ),
            bool(payload["conversation_state"].get("catalog_revision")),
        )
        if all(checks):
            passed += 1
        else:
            failures.append(f"{query}: trajectory gate failed")

    state = None
    stable_codes: list[str] | None = None
    dialogue = (
        ("Nên chọn Dell hay Asus tầm 20 triệu?", "product-comparison"),
        ("Ưu tiên độ bền và hiệu năng.", "product-comparison"),
        ("Máy nào đáng tiền hơn?", "product-comparison"),
        ("Tại sao Dell lại rẻ hơn?", "price-causality"),
        ("Nếu chơi game thì sao?", "product-comparison"),
    )
    for query, expected_skill in dialogue:
        payload = client.post(
            "/api/chat",
            json={
                "message": query,
                "history": [],
                "conversation_state": state,
            },
            headers={"x-eval-mode": "harness"},
        ).json()
        state = payload["conversation_state"]
        trace = payload["decision_trace"]["harness"]
        codes = state["candidate_codes"]
        if stable_codes is None:
            stable_codes = codes
        checks = (
            trace["skill"]["name"] == expected_skill,
            state["category"] == "Laptop",
            state["budget_target"] == 20_000_000,
            state["compared_brands"] == ["Dell", "Asus"],
            codes == stable_codes,
        )
        if all(checks):
            passed += 1
        else:
            failures.append(f"multi-turn {query}: context trajectory drifted")

    total = len(SCENARIOS) + len(dialogue)
    print(f"Harness dynamic gates: {passed}/{total}")
    for failure in failures:
        print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
