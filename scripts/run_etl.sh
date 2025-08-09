#!/bin/bash

# Coal Mining ETL Pipeline Runner
# This script runs the ETL pipeline manually

set -e

echo "🔄 Running Coal Mining ETL Pipeline..."

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Services are not running. Please start with: docker-compose up -d"
    exit 1
fi

# Run ETL pipeline
echo "📊 Extracting, transforming, and loading data..."
docker-compose exec etl_runner python etl_pipeline.py

# Run data validation
echo "🔍 Running data validation checks..."
docker-compose exec etl_runner python data_validation.py

# Display summary
echo ""
echo "✅ ETL Pipeline completed successfully!"
echo ""
echo "📈 View results in Metabase: http://localhost:3000"
echo "📄 Check validation report: logs/validation_report.json"
echo "📋 View ETL logs: logs/etl_pipeline.log"
