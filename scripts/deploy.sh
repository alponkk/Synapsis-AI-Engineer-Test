#!/bin/bash

# Coal Mining Data Pipeline Deployment Script
# This script deploys the complete coal mining data pipeline

set -e

echo "🚀 Starting Coal Mining Data Pipeline Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p clickhouse_data
mkdir -p metabase_data

# Stop any existing services
echo "🔄 Stopping any existing services..."
docker-compose down -v

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose pull

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose up -d --build

echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check ClickHouse
if docker-compose exec -T clickhouse clickhouse-client -q "SELECT 1" &> /dev/null; then
    echo "✅ ClickHouse is healthy"
else
    echo "❌ ClickHouse is not responding"
    exit 1
fi

# Check Metabase (may take longer to start)
echo "⏳ Waiting for Metabase to start (this may take a few minutes)..."
for i in {1..20}; do
    if curl -s http://localhost:3000/api/health &> /dev/null; then
        echo "✅ Metabase is healthy"
        break
    else
        echo "   Attempt $i/20 - Metabase still starting..."
        sleep 15
    fi
done

# Run ETL pipeline
echo "🔄 Running initial ETL pipeline..."
docker-compose exec -T etl_runner python etl_pipeline.py

# Run data validation
echo "🔍 Running data validation..."
docker-compose exec -T etl_runner python data_validation.py

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Access your dashboard at: http://localhost:3000"
echo "🔑 Login credentials:"
echo "   Email: admin@coalmining.com"
echo "   Password: CoalMining123!"
echo ""
echo "🔧 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo ""
echo "📖 For more information, see README.md"
