# Verification configuration

Thư mục này có hai cơ chế cấu hình độc lập. Chọn đúng cơ chế cho phần đang
thay đổi; chúng không tự động đồng bộ giá trị với nhau.

| Mục đích | Entry point | Nguồn YAML |
| --- | --- | --- |
| Cấu hình workflow/prompt ở dạng `dict` | `config_loader.py` | `verification_config.yaml`, `workflow_config.yaml`, `prompts.yaml`, `environments/<environment>.yaml` |
| Ngưỡng severity có Pydantic validation | `thresholds_config.py` | `thresholds.yaml` |

## Loader YAML chung

`ConfigLoader` đọc file gốc từ `base_path` (mặc định là working directory),
deep-merge file môi trường khi tên file tồn tại, rồi trả về dictionary. Các tên
môi trường có sẵn cho loader là `development`, `production` và `test`.

```python
from backend.verification.config.config_loader import ConfigLoader

loader = ConfigLoader(base_path=".")
configs = loader.load_all_configs("development")

price_tolerance = configs["verification"]["price_accuracy"]["tolerance_percent"]
workflow_timeout = configs["workflow"]["workflow"]["execution"]["max_execution_time_seconds"]
```

Các hàm tiện dụng là `load_config_for_environment()`,
`validate_all_configs()` và `get_config_loader()`. `validate_config()` chỉ kiểm
tra cấu trúc tối thiểu của từng loại config, không phải schema đầy đủ của toàn
bộ YAML. `clear_cache()`/`reload()` xoá snapshot nội bộ rồi đọc lại file; đừng
coi chúng là file watcher hay hot reload tự động.

### Biến môi trường cho loader

Những biến dưới đây chỉ áp dụng khi gọi `load_verification_config()` (hoặc
`load_all_configs()`); giá trị boolean đúng là `true`, `1`, `yes` hoặc `on`.

| Biến | Đường dẫn được ghi đè | Kiểu |
| --- | --- | --- |
| `VERIFICATION_PRICE_TOLERANCE` | `price_accuracy.tolerance_percent` | float |
| `VERIFICATION_MAX_RETRIES` | `retry_settings.max_retries` | int |
| `VERIFICATION_PARALLEL_CHECKS` | `performance.parallel_verification` | bool |
| `VERIFICATION_LOG_LEVEL` | `logging.level` | string |
| `VERIFICATION_CACHE_ENABLED` | `performance.caching.enabled` | bool |
| `VERIFICATION_TIMEOUT` | `performance.verification_timeout_seconds` | int |

`VERIFICATION_ENVIRONMENT` (ưu tiên) hoặc `ENVIRONMENT` chỉ được dùng để chọn
môi trường khi không truyền đối số `environment`.

## Pydantic config

`ConfigLoader.get_pydantic_config()` trả về `VerificationConfig` từ
`config.py`. Dùng API này khi consumer cần model đã validate thay vì dict:

```python
from backend.verification.config.config_loader import ConfigLoader

config = ConfigLoader(base_path=".").get_pydantic_config("production")
print(config.max_retries)
```

## Threshold configuration

Ngưỡng price/policy/relevance/escalation có tài liệu chi tiết tại
[`THRESHOLDS_README.md`](THRESHOLDS_README.md). Ví dụ có thể chạy từ root:

```powershell
python tools/examples/config_example_usage.py
python tools/examples/thresholds_example.py
pytest -q tests/verification/test_config_loader.py tests/verification/test_thresholds_config.py
```

Không đặt secret vào bất kỳ YAML nào trong thư mục này. Dùng biến môi trường
hoặc secret store của môi trường deploy cho token và credential.
