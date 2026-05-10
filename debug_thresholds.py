"""Debug script to test thresholds configuration"""

print("Starting debug...")

try:
    print("Importing enum...")
    from enum import Enum
    print("Enum imported successfully")
    
    print("Importing typing...")
    from typing import Dict, List, Optional, Literal, Any
    print("Typing imported successfully")
    
    print("Importing pydantic...")
    from pydantic import BaseModel, Field, validator
    print("Pydantic imported successfully")
    
    print("Defining IssueSeverity...")
    class IssueSeverity(str, Enum):
        """Issue severity levels cho structured classification"""
        CRITICAL = "critical"
        MAJOR = "major"
        MINOR = "minor"
    
    print("IssueSeverity defined successfully:", IssueSeverity.CRITICAL)
    
    print("Defining PriceAccuracyThresholds...")
    class PriceAccuracyThresholds(BaseModel):
        """Price accuracy thresholds"""
        minor_threshold_percent: float = Field(default=5.0, ge=0.0, le=100.0)
        major_threshold_percent: float = Field(default=15.0, ge=0.0, le=100.0)
        critical_threshold_percent: float = Field(default=30.0, ge=0.0, le=100.0)
        pass_tolerance_percent: float = Field(default=1.0, ge=0.0, le=100.0)
        missing_price_severity: IssueSeverity = Field(default=IssueSeverity.MAJOR)
    
    print("PriceAccuracyThresholds defined successfully")
    
    # Test instantiation
    thresholds = PriceAccuracyThresholds()
    print("Thresholds created:", thresholds.minor_threshold_percent)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()