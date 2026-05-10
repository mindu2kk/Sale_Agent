"""
Test Suite cho Verification Agent

Comprehensive testing với:
- Unit tests cho individual components
- Property-based tests với Hypothesis
- Integration tests cho end-to-end workflows
- Performance tests cho optimization validation
"""

# Test configuration
TEST_CONFIG = {
    "price_tolerance_percent": 1.0,
    "max_retries": 2,  # Reduced for faster testing
    "parallel_verification": True,
    "early_termination": True,
    "async_timeout_seconds": 10,  # Reduced for faster testing
    "enable_caching": False,  # Disabled for test isolation
    "log_level": "WARNING"  # Reduced logging for cleaner test output
}

# Test data constants
SAMPLE_OBJECTIONS = [
    "iPhone 15 Pro Max có giá bao nhiêu?",
    "Chính sách bảo hành của Samsung như thế nào?",
    "So sánh iPhone vs Samsung Galaxy về camera",
    "Tôi muốn đổi trả sản phẩm được không?"
]

SAMPLE_DRAFT_RESPONSES = [
    "iPhone 15 Pro Max có giá 34,990,000 VND với bảo hành 1 năm chính hãng.",
    "Samsung cung cấp bảo hành 2 năm cho tất cả smartphone và có chính sách đổi trả trong 30 ngày.",
    "iPhone có camera tốt hơn Samsung về chụp đêm, nhưng Samsung có zoom xa hơn.",
    "Bạn có thể đổi trả sản phẩm trong vòng 15 ngày kể từ ngày mua với điều kiện còn nguyên seal."
]