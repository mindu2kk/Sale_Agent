import sys
sys.path.insert(0, '.')

try:
    from verification.config.thresholds_config import IssueSeverity, PriceAccuracyThresholds
    print("SUCCESS: Classes imported successfully")
    
    # Test enum
    print(f"IssueSeverity.CRITICAL = {IssueSeverity.CRITICAL}")
    
    # Test class instantiation
    thresholds = PriceAccuracyThresholds()
    print(f"Default minor threshold: {thresholds.minor_threshold_percent}%")
    
    # Test classification
    severity = thresholds.classify_price_deviation(10.0)
    print(f"10% deviation classified as: {severity}")
    
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()