#!/usr/bin/env python3
"""
Test script for forecasting pipeline to verify installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import scikit-learn: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import seaborn: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✓ xgboost imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import xgboost: {e}")
        return False
    
    try:
        from sqlalchemy import create_engine
        print("✓ sqlalchemy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sqlalchemy: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'forecasting/models',
        'forecasting/results', 
        'forecasting/visualizations',
        'forecasting/data'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} does not exist")
            all_exist = False
    
    return all_exist

def test_database_connection():
    """Test database connection"""
    print("\nTesting database connection...")
    
    try:
        from sqlalchemy import create_engine, text
        
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = int(os.getenv('POSTGRES_PORT', 5432))
        pg_user = os.getenv('POSTGRES_USER', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        pg_database = os.getenv('POSTGRES_DB', 'coal_mining')
        
        connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        print("✓ Database connection successful")
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Forecasting Pipeline Test ===\n")
    
    imports_ok = test_imports()
    dirs_ok = test_directories()
    db_ok = test_database_connection()
    
    print("\n=== Test Summary ===")
    if imports_ok and dirs_ok and db_ok:
        print("✓ All tests passed! Forecasting pipeline is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
