"""Production-oriented regression gate for realistic shopping conversations."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.main import app


SINGLE_TURN_CASES = (
    ("Tư vấn laptop tầm 20 triệu", "Laptop", None),
    ("Có MacBook nào khoảng 20 triệu không?", "Laptop", "Apple"),
    ("Laptop Dell dưới 25 triệu để làm văn phòng", "Laptop", "Dell"),
    ("Laptop Asus chơi game có card rời tầm 22 triệu", "Laptop", "Asus"),
    ("Điện thoại Samsung tầm 10 triệu chụp ảnh tốt", "Mobile Phone", "Samsung"),
    ("iPhone nào rẻ nhất trong catalog?", "Mobile Phone", "Apple"),
    ("Máy MSI nào có RTX 4050?", "Laptop", "MSI"),
    ("Tư vấn chi tiết mã 00928595", "Laptop", "HP"),
    ("Có điện thoại Oppo dưới 5 triệu không?", "Mobile Phone", "Oppo"),
    ("Laptop Lenovo khoảng 15 triệu", "Laptop", "Lenovo"),
    ("Laptop Acer cho sinh viên lập trình", "Laptop", "Acer"),
    ("Có laptop HP 14 inch không?", "Laptop", "HP"),
)

COMPARISON_CASES = (
    (
        "So sánh MSI Gaming Thin 15 B13UC-3247VN, "
        "Dell 15 DC15255 R7-7730U và HP 14-ep1179TU Core 5",
        {"00921510", "00927423", "00921548"},
    ),
    (
        "So sánh HP 14-ep1179TU Core 5 với Dell 15 DC15255 R7-7730U",
        {"00921548", "00927423"},
    ),
)


def _ask(client: TestClient, message: str, state: dict | None = None) -> dict:
    response = client.post(
        "/api/chat",
        json={"message": message, "history": [], "conversation_state": state},
    )
    response.raise_for_status()
    return response.json()


def run() -> dict:
    client = TestClient(app)
    checks: list[dict] = []
    latencies: list[float] = []

    for message, category, brand in SINGLE_TURN_CASES:
        started = time.perf_counter()
        payload = _ask(client, message)
        latencies.append(time.perf_counter() - started)
        products = payload["products"]
        checks.append(
            {
                "case": message,
                "passed": bool(products)
                and all(product["category"] == category for product in products)
                and (
                    brand is None
                    or all(product["brand"] == brand for product in products)
                ),
            }
        )

    for message, expected_codes in COMPARISON_CASES:
        payload = _ask(client, message)
        checks.append(
            {
                "case": message,
                "passed": payload["answer_type"] == "comparison"
                and payload["verification"]["approved"] is True
                and {product["code"] for product in payload["products"]}
                == expected_codes,
            }
        )

    first = _ask(client, "Tư vấn laptop tầm 20 triệu")
    macbook = _ask(
        client,
        "Có MacBook trong tầm giá không?",
        first["conversation_state"],
    )
    checks.append(
        {
            "case": "Inherited budget plus MacBook brand constraint",
            "passed": macbook["active_context"]["budget_target"] == 20_000_000
            and bool(macbook["products"])
            and all(
                product["brand"] == "Apple"
                and product["category"] == "Laptop"
                for product in macbook["products"]
            ),
        }
    )

    comparison = _ask(client, "Nên chọn Dell hay Asus tầm 20 triệu?")
    expected_candidates = set(comparison["active_context"]["candidate_codes"])
    state = comparison["conversation_state"]
    for follow_up in (
        "Ưu tiên độ bền và hiệu năng",
        "Máy nào đáng tiền hơn?",
        "Nếu chơi game thì sao?",
    ):
        payload = _ask(client, follow_up, state)
        checks.append(
            {
                "case": f"Context preservation: {follow_up}",
                "passed": payload["active_context"]["budget_target"] == 20_000_000
                and set(payload["active_context"]["candidate_codes"])
                == expected_candidates
                and {product["brand"] for product in payload["products"]}
                <= {"Dell", "Asus"},
            }
        )
        state = payload["conversation_state"]

    switch = _ask(client, "Chuyển sang điện thoại tầm 8 triệu", state)
    checks.append(
        {
            "case": "Explicit category switch",
            "passed": switch["active_context"]["category"] == "Mobile Phone"
            and all(
                product["category"] == "Mobile Phone"
                for product in switch["products"]
            ),
        }
    )

    passed = sum(1 for check in checks if check["passed"])
    sorted_latency = sorted(latencies)
    report = {
        "cases": len(checks),
        "passed": passed,
        "rate": round(passed / len(checks), 4),
        "p95_seconds": round(
            sorted_latency[max(0, int(len(sorted_latency) * 0.95) - 1)],
            4,
        ),
        "failures": [check["case"] for check in checks if not check["passed"]],
    }
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["rate"] == 1.0 and result["p95_seconds"] < 6 else 1)
