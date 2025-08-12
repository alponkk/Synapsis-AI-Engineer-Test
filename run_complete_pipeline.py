#!/usr/bin/env python3
"""
Complete Pipeline Runner - ETL + Forecasting
Author: Data Engineering Team
Description: Run the complete data pipeline from ETL to forecasting
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and return success status"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Run the complete pipeline"""
    logger.info("=== Starting Complete Data Pipeline ===")
    start_time = datetime.now()
    
    # Step 1: Run ETL Pipeline
    logger.info("Step 1: Running ETL Pipeline")
    if not run_command("python python_scripts/etl_pipeline.py", "ETL Pipeline"):
        logger.error("ETL Pipeline failed. Stopping execution.")
        return 1
    
    # Step 2: Run Data Validation
    logger.info("Step 2: Running Data Validation")
    if not run_command("python python_scripts/data_validation.py", "Data Validation"):
        logger.warning("Data Validation failed, but continuing with forecasting")
    
    # Step 3: Run Forecasting
    logger.info("Step 3: Running Production Forecasting")
    if not run_command("python forecasting/production_forecasting.py", "Production Forecasting"):
        logger.error("Forecasting Pipeline failed")
        return 1
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    logger.info("=== Complete Pipeline Finished Successfully! ===")
    logger.info(f"Total execution time: {execution_time}")
    logger.info("Generated outputs:")
    logger.info("  - ETL: Updated daily_production_metrics table")
    logger.info("  - Forecasting Models: forecasting/models/")
    logger.info("  - Visualizations: forecasting/visualizations/")
    logger.info("  - Results: forecasting/results/")
    
    return 0

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the complete pipeline
    sys.exit(main())
