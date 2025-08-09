# Quick Deployment Guide

## 🚀 One-Command Deployment

```bash
# For Linux/macOS
./scripts/deploy.sh

# For Windows (PowerShell)
docker-compose up -d --build
```

## 📋 Prerequisites

- Docker Desktop installed and running
- 8GB+ RAM available
- Ports 3000, 8123, 9000 available

## ⚡ Quick Start Steps

1. **Clone Repository**
   ```bash
   git clone <your-repository-url>
   cd coal-mining-pipeline
   ```

2. **Deploy Services**
   ```bash
   docker-compose up -d --build
   ```

3. **Wait for Initialization** (2-3 minutes)
   
4. **Access Dashboard**
   - URL: http://localhost:3000
   - Email: admin@coalmining.com  
   - Password: CoalMining123!

## 🔧 Management Commands

```bash
# View system status
docker-compose ps

# Run ETL manually
docker-compose exec etl_runner python etl_pipeline.py

# Run data validation
docker-compose exec etl_runner python data_validation.py

# Setup Metabase dashboards
docker-compose exec etl_runner python metabase_setup.py

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart
```

## 📊 Expected Results

After successful deployment, you should see:

1. **ClickHouse Database** with tables:
   - `mines` (3 records)
   - `production_logs` (2000+ records)
   - `equipment_sensors` (40,000+ records)
   - `weather_data` (daily records)
   - `daily_production_metrics` (aggregated metrics)

2. **Metabase Dashboard** with 9 visualizations:
   - Daily Production Trend
   - Average Quality Grade
   - Equipment Utilization
   - Fuel Efficiency
   - Weather Impact Analysis
   - Production by Mine
   - Monthly Production Summary
   - Equipment Status Distribution
   - Production vs Weather Correlation

3. **Data Validation Reports** in `logs/` directory

## 🛠️ Troubleshooting

**Services won't start:**
```bash
# Check Docker is running
docker --version

# Free up resources
docker system prune

# Restart Docker Desktop
```

**ETL fails:**
```bash
# Check data files exist
ls -la "synapsis ai engineer challege datasets/"

# Check ClickHouse logs
docker-compose logs clickhouse
```

**Dashboard doesn't load:**
```bash
# Wait longer (Metabase takes 2-3 minutes)
# Check if service is healthy
curl http://localhost:3000/api/health
```

## 📈 Data Flow Verification

1. **Check Production Data:**
   ```sql
   SELECT COUNT(*) FROM production_logs;
   -- Expected: 2000+ records
   ```

2. **Check Sensor Data:**
   ```sql
   SELECT COUNT(*) FROM equipment_sensors;
   -- Expected: 40,000+ records
   ```

3. **Check Metrics:**
   ```sql
   SELECT COUNT(*) FROM daily_production_metrics;
   -- Expected: 100+ daily records
   ```

## 🎯 Success Criteria

✅ All 4 Docker containers running  
✅ ClickHouse accepting connections on port 8123  
✅ Metabase accessible on port 3000  
✅ ETL pipeline completes without errors  
✅ Data validation passes with quality score >90  
✅ All 9 dashboard visualizations displaying data  

## 🆘 Support

1. Check `README.md` for detailed information
2. Review `Coal_Mining_Data_Pipeline_Report.md` for technical details
3. Examine logs in the `logs/` directory
4. Use monitoring script: `./scripts/monitor.sh` (Linux/macOS)

---

**Deployment Time**: ~5 minutes  
**Manual Setup Required**: None (fully automated)  
**Default Credentials**: admin@coalmining.com / CoalMining123!
