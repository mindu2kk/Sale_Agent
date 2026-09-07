"""Trace-level harness evaluation derived from paper Section 6.6."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["EXPOSE_DECISION_TRACE"] = "true"

from backend.api.main import app


SCENARIOS = (
    ("Tư vấn laptop tầm 20 triệu", "Laptop"),
    ("Có MacBook trong tầm giá không?", "Laptop"),
    ("So sánh Oppo A6C 4GB, Tecno Spark 50 4GB", "Mobile Phone"),
    ("So sánh Dell với Asus tầm 20 triệu", "Laptop"),
    ("Điện thoại Samsung tầm 8 triệu", "Mobile Phone"),
    ("Laptop có card rời tầm 25 triệu", "Laptop"),
    ("Tư vấn chi tiết mã 00928595", "Laptop"),
    ("Có mẫu nào rẻ hơn 00928595 không?", "Laptop"),
    ("Apple bảo hành bao lâu?", None),
    ("Tư vấn máy cho tôi", None),
)


def run() -> dict:
    client = TestClient(app)
    checks: list[dict] = []
    latencies: list[float] = []

    for message, expected_category in SCENARIOS:
        started = time.perf_counter()
        response = client.post(
            "/api/chat",
            json={
                "message": message,
                "history": [],
                "conversation_state": None,
            },
            headers={"x-eval-mode": "harness"},
        )
        latencies.append((time.perf_counter() - started) * 1000)
        payload = response.json()
        trace = payload.get("decision_trace", {}).get("harness", {})
        events = trace.get("events", [])
        phases = [event.get("phase") for event in events]
        answer_codes = {
            product["code"] for product in payload.get("products", [])
        }
        evidence_codes = {
            item.get("product_code")
            for item in trace.get("evidence", [])
            if item.get("product_code")
        }
        failed_events = [
            event for event in events if event.get("status") == "failed"
        ]
        recovered = any(
            event.get("phase") == "recovery" for event in events
        )
        expected_recovery = payload["answer_type"] in {
            "clarify",
            "safe_degrade",
        }

        checks.append(
            {
                "scenario": message,
                "terminal": trace.get("terminal_status") is not None,
                "phase_order": phases[:2] == ["perception", "planning"]
                and "execution" in phases
                and "verification" in phases
                and phases[-1:] == ["commit"],
                "state": expected_category is None
                or trace.get("belief", {}).get("category")
                == expected_category,
                "evidence": answer_codes <= evidence_codes,
                "governance": not failed_events or recovered,
                "recovery": recovered == expected_recovery
                or payload["ai_mode"] == "catalog_fallback",
                "budget": trace.get("elapsed_ms", 99_999) < 6_000,
            }
        )

    metric_names = (
        "terminal",
        "phase_order",
        "state",
        "evidence",
        "governance",
        "recovery",
        "budget",
    )
    metrics = {
        name: {
            "passed": sum(bool(item[name]) for item in checks),
            "total": len(checks),
        }
        for name in metric_names
    }
    for metric in metrics.values():
        metric["rate"] = round(metric["passed"] / metric["total"], 4)

    return {
        "cases": len(checks),
        "metrics": metrics,
        "p95_ms": round(
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
            2,
        ),
        "failures": [
            {
                "scenario": item["scenario"],
                "failed_metrics": [
                    name for name in metric_names if not item[name]
                ],
            }
            for item in checks
            if any(not item[name] for name in metric_names)
        ],
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    hard_failure = any(
        metric["rate"] < 1.0
        for metric in report["metrics"].values()
    )
    raise SystemExit(1 if hard_failure or report["p95_ms"] >= 6_000 else 0)
