try:
    import verification.config.thresholds_config as tc
    print("Import successful")
    print("Available classes:", [name for name in dir(tc) if not name.startswith('_')])
    
    # Try to import specific classes
    from verification.config.thresholds_config import IssueSeverity
    print("IssueSeverity imported successfully")
    
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()