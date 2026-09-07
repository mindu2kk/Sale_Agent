# Verification thresholds

`thresholds_config.py` là cấu hình Pydantic cho việc phân loại issue của
verification layer. Nó tách biệt với `config_loader.py`; các biến
`VERIFICATION_*` của loader chung không thay đổi `VerificationThresholdsConfig`.

## API chính

```python
from backend.verification.config.thresholds_config import (
    enhanced_load_thresholds_config,
)

config = enhanced_load_thresholds_config(
    "backend/verification/config/thresholds.yaml"
)
severity = config.price_accuracy.classify_price_deviation(8.0)
is_pass = config.price_accuracy.should_pass_price_check(0.8)
```

Các model công khai:

- `PriceAccuracyThresholds`: phân loại chênh lệch giá và kiểm tra pass.
- `PolicyAuthenticityThresholds`: phân loại policy fabricated/inaccurate/
  incomplete hoặc thiếu citation.
- `TopicRelevanceThresholds`: phân loại coverage/off-topic và kiểm tra pass.
- `EscalationThresholds`: quyết định escalation, early termination và retry.
- `TimeoutConfig`: timeout theo tên operation.
- `VerificationThresholdsConfig`: aggregate các model trên.

`get_default_thresholds_config()` trả về defaults trong code.
`enhanced_load_thresholds_config(path)` đọc YAML nếu `path` tồn tại, áp dụng
biến môi trường, rồi validate thành Pydantic model. File không tồn tại không
raise; nó trả về defaults (và các override từ environment). Dùng
`save_thresholds_config(config, path)` để lưu model hợp lệ thành YAML phẳng
theo field Pydantic.

## Quy tắc đang được thực thi

Với default hiện tại:

| Kiểm tra | Kết quả |
| --- | --- |
| Chênh lệch giá `<= 1%` | price check pass |
| Chênh lệch giá `< 5%`, `5%.. <30%`, `>=30%` | minor, major, critical |
| Policy fabricated | critical và không pass |
| Policy inaccurate hoặc thiếu citation bắt buộc | không pass |
| Coverage `<50%` hoặc off-topic `>30%` | critical relevance issue |
| Coverage `50%.. <80%` | major relevance issue |
| Coverage `>=70%`; nếu có empathy score thì score `>=0.5` | relevance check pass |
| fabricated policy, critical price deviation, completely irrelevant, hoặc count vượt ngưỡng | escalate ngay |

Lưu ý: `classify_relevance_issue()` hiện trả về `critical` cho mọi coverage
dưới `major_coverage_threshold` (mặc định `0.5`).

`should_escalate_immediately()` dùng điều kiện **lớn hơn** ngưỡng count, nên
default `max_critical_issues_before_escalation=2` escalate từ 3 critical
issues. `should_terminate_early()` là rule khác: mặc định terminate từ 3
critical issues khi `early_termination_enabled=True`.

## YAML được map như thế nào

`thresholds.yaml` có nhiều section phục vụ tài liệu/ý định. Loader chỉ map các
key dưới đây; key khác không làm thay đổi Pydantic model hiện tại:

| YAML section | Field được map |
| --- | --- |
| `price_accuracy.thresholds` | các `*_threshold_percent` |
| `price_accuracy.pass_criteria.tolerance_percent` | `pass_tolerance_percent` |
| `policy_authenticity.severity_rules` | 4 severity fields |
| `policy_authenticity.citation_requirements.citation_required` | `citation_required` |
| `topic_relevance.coverage_thresholds` | coverage threshold fields |
| `topic_relevance.pass_criteria.min_coverage_ratio` | `pass_coverage_threshold` |
| `topic_relevance.empathy_requirements` | `empathy_required`, `min_empathy_score` |
| `escalation.count_thresholds` | max critical/major/total before escalation |
| `escalation.immediate_triggers` | fabricated policy và critical price deviation |
| `escalation.early_termination` | enabled, stop-on-first-critical, multiple-critical threshold |
| `escalation.retry_limits` | ba retry limit theo severity |
| `timeouts` | matching `TimeoutConfig` fields |

Môi trường cho threshold model được chọn bằng code:

```python
base = enhanced_load_thresholds_config("backend/verification/config/thresholds.yaml")
development = base.get_environment_config("development")
production = base.get_environment_config("production")
testing = base.get_environment_config("testing")
```

Đừng dựa vào section `environments:` trong `thresholds.yaml` cho API trên:
`get_environment_config()` dùng override đã định nghĩa trong
`thresholds_config.py`.

### Environment overrides cho threshold model

Chỉ ba biến này được `enhanced_load_thresholds_config()` đọc:

| Biến | Field |
| --- | --- |
| `VERIFICATION_THRESHOLDS_PRICE_TOLERANCE` | `price_accuracy.pass_tolerance_percent` |
| `VERIFICATION_THRESHOLDS_ESCALATION_MAX_CRITICAL` | `escalation.max_critical_issues_before_escalation` |
| `VERIFICATION_THRESHOLDS_EARLY_TERMINATION` | `escalation.early_termination_enabled` (`true` là bật) |

## Kiểm tra thay đổi

```powershell
python tools/examples/thresholds_example.py
pytest -q tests/verification/test_thresholds_config.py
```

Ví dụ lưu cấu hình dùng `TemporaryDirectory`, nên không tạo YAML mới trong
repository. Sau khi thay đổi field hoặc mapping YAML, thêm test cho mapping và
giữ tổng `verification_weights` bằng `1.0` (sai lệch tối đa `0.01`).
