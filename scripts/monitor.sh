#!/bin/bash

# Coal Mining Pipeline Monitoring Script
# This script provides system monitoring and health checks

set -e

echo "🔍 Coal Mining Pipeline Health Check"
echo "===================================="

# Check Docker services
echo ""
echo "📦 Docker Services Status:"
docker-compose ps

# Check service health
echo ""
echo "🏥 Service Health Checks:"

# ClickHouse health
if docker-compose exec -T clickhouse clickhouse-client -q "SELECT 1" &> /dev/null; then
    echo "✅ ClickHouse: Healthy"
    
    # Get database stats
    echo "   📊 Database Statistics:"
    docker-compose exec -T clickhouse clickhouse-client -q "
        SELECT 
            table AS 'Table',
            formatReadableSize(total_bytes) AS 'Size',
            rows AS 'Rows'
        FROM system.tables 
        WHERE database = 'coal_mining' 
        AND engine != 'SystemLog'
        FORMAT PrettyCompact
    "
else
    echo "❌ ClickHouse: Unhealthy"
fi

# Metabase health
if curl -s http://localhost:3000/api/health &> /dev/null; then
    echo "✅ Metabase: Healthy"
else
    echo "❌ Metabase: Unhealthy"
fi

# Check recent logs for errors
echo ""
echo "📋 Recent Log Summary:"
if [ -f "logs/etl_pipeline.log" ]; then
    error_count=$(grep -c "ERROR" logs/etl_pipeline.log 2>/dev/null || echo "0")
    warning_count=$(grep -c "WARNING" logs/etl_pipeline.log 2>/dev/null || echo "0")
    echo "   🔥 ETL Errors: $error_count"
    echo "   ⚠️ ETL Warnings: $warning_count"
    
    if [ "$error_count" -gt 0 ]; then
        echo "   📝 Recent Errors:"
        tail -n 20 logs/etl_pipeline.log | grep "ERROR" | tail -n 5
    fi
else
    echo "   📄 No ETL logs found"
fi

# Check data freshness
echo ""
echo "📅 Data Freshness:"
if docker-compose exec -T clickhouse clickhouse-client -q "SELECT 1" &> /dev/null; then
    latest_date=$(docker-compose exec -T clickhouse clickhouse-client -q "
        SELECT max(date) FROM daily_production_metrics FORMAT TSV
    " 2>/dev/null || echo "No data")
    echo "   📊 Latest Production Data: $latest_date"
    
    record_count=$(docker-compose exec -T clickhouse clickhouse-client -q "
        SELECT count() FROM daily_production_metrics FORMAT TSV
    " 2>/dev/null || echo "0")
    echo "   📈 Total Metrics Records: $record_count"
fi

# System resources
echo ""
echo "💾 System Resources:"
echo "   🖥️ Docker Stats:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Data quality summary
echo ""
echo "🔍 Data Quality Summary:"
if [ -f "logs/validation_report.json" ]; then
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('logs/validation_report.json', 'r') as f:
        report = json.load(f)
    assessment = report.get('overall_assessment', {})
    print(f\"   📊 Average Quality Score: {assessment.get('average_quality_score', 'N/A'):.1f}\")
    print(f\"   🏥 Data Health: {assessment.get('data_health', 'N/A')}\")
    print(f\"   🚨 Critical Issues: {assessment.get('critical_issues', 'N/A')}\")
    print(f\"   ⚠️ Total Anomalies: {assessment.get('total_anomalies', 'N/A')}\")
except:
    print('   📄 Validation report not available')
"
    else
        echo "   📄 Python not available for report parsing"
    fi
else
    echo "   📄 No validation report found"
fi

echo ""
echo "🔗 Quick Links:"
echo "   📊 Dashboard: http://localhost:3000"
echo "   🗄️ ClickHouse: http://localhost:8123"
echo ""
echo "📝 Log Files:"
echo "   📋 ETL Logs: logs/etl_pipeline.log"
echo "   🔍 Validation Logs: logs/data_validation.log"
echo "   📊 Validation Report: logs/validation_report.json"
