# Coal Mining Data Pipeline

A comprehensive data engineering solution for optimizing coal mining operations through automated data collection, transformation, and visualization.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended

### Deployment
```bash
# Clone the repository
git clone <repository-url>
cd coal-mining-pipeline

# Start all services
docker-compose up -d

# Wait for services to initialize (2-3 minutes)
# Access Metabase dashboard at http://localhost:3000
```

### Default Credentials
- **Metabase Login**: admin@coalmining.com
- **Password**: CoalMining123!

## 📊 System Overview

This pipeline processes data from three sources:
1. **SQL Database**: Production logs with mining operations data
2. **CSV Files**: Equipment sensor readings (IoT data)
3. **Weather API**: Daily weather data for Berau, Kalimantan

### Key Metrics Generated
- **Total Production Daily**: Daily coal extraction totals
- **Average Quality Grade**: Coal quality assessment
- **Equipment Utilization**: Operational efficiency percentages
- **Fuel Efficiency**: Fuel consumption per ton of coal
- **Weather Impact**: Correlation between weather and production

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Sources│───▶│ ETL Pipeline│───▶│ ClickHouse  │───▶│  Metabase   │
│ • SQL       │    │ • Extract   │    │ Data        │    │ Dashboard   │
│ • CSV       │    │ • Transform │    │ Warehouse   │    │             │
│ • Weather   │    │ • Load      │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                    ▲
                           ▼                    │
                   ┌─────────────┐              │
                   │ Data        │──────────────┘
                   │ Validation  │
                   └─────────────┘
```

## 📁 Project Structure

```
coal-mining-pipeline/
├── docker-compose.yml              # Main orchestration file
├── Dockerfile.etl                  # ETL container definition
├── requirements.txt                # Python dependencies
├── etl_pipeline.py                 # Main ETL script
├── data_validation.py              # Data quality validation
├── metabase_setup.py              # Dashboard configuration
├── clickhouse-config/             # ClickHouse configuration
│   └── users.xml
├── init-scripts/                  # Database initialization
│   └── 01_create_database.sql
├── synapsis ai engineer challege datasets/
│   ├── production_logs.sql        # Production data
│   └── equipment_sensors.csv      # Sensor data
├── logs/                          # Application logs
└── Coal_Mining_Data_Pipeline_Report.md
```

## 🔧 Configuration

### Environment Variables
```bash
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=password
CLICKHOUSE_DB=coal_mining
```

### Service Ports
- **ClickHouse HTTP**: 8123
- **ClickHouse Native**: 9000
- **Metabase**: 3000

## 📈 Dashboard Features

### Available Visualizations
1. **Daily Production Trend** - Production over time
2. **Quality Grade Analysis** - Coal quality metrics
3. **Equipment Utilization** - Operational efficiency
4. **Fuel Efficiency Tracking** - Resource optimization
5. **Weather Impact Analysis** - Environmental correlations
6. **Mine Comparison** - Performance by location
7. **Monthly Summaries** - Aggregated insights
8. **Equipment Status** - Real-time monitoring
9. **Weather Correlation** - Production vs weather patterns

## 🔍 Data Validation

### Automated Checks
- ✅ **Range Validation**: Ensures values within expected bounds
- ✅ **Null Detection**: Identifies missing data
- ✅ **Duplicate Prevention**: Prevents data duplication
- ✅ **Anomaly Detection**: Statistical outlier identification
- ✅ **Quality Scoring**: Overall data health assessment

### Quality Metrics
- **Production Rules**: No negative extraction values
- **Utilization Rules**: 0-100% equipment utilization
- **Quality Rules**: 0-10 coal quality grade scale
- **Weather Rules**: Complete daily weather data

## 🛠️ Operations

### Running ETL Manually
```bash
# Execute ETL pipeline
docker-compose exec etl_runner python etl_pipeline.py

# Run data validation
docker-compose exec etl_runner python data_validation.py

# Setup Metabase dashboards
docker-compose exec etl_runner python metabase_setup.py
```

### Viewing Logs
```bash
# ETL logs
docker-compose logs etl_runner

# ClickHouse logs
docker-compose logs clickhouse

# All service logs
docker-compose logs -f
```

### Database Access
```bash
# Connect to ClickHouse
docker-compose exec clickhouse clickhouse-client

# Example queries
SELECT * FROM daily_production_metrics LIMIT 10;
SELECT COUNT(*) FROM production_logs;
```

## 📊 Sample Queries

### Production Analysis
```sql
-- Daily production summary
SELECT 
    date,
    total_production_daily,
    average_quality_grade,
    equipment_utilization
FROM daily_production_metrics
ORDER BY date DESC;
```

### Weather Impact Analysis
```sql
-- Weather correlation
SELECT 
    precipitation_sum,
    AVG(total_production_daily) as avg_production
FROM daily_production_metrics
GROUP BY precipitation_sum
ORDER BY precipitation_sum;
```

## 🚨 Troubleshooting

### Common Issues

**ETL Pipeline Fails**
```bash
# Check data source files
ls -la "synapsis ai engineer challege datasets/"

# Verify ClickHouse connectivity
docker-compose exec etl_runner python -c "import clickhouse_connect; print('OK')"
```

**Metabase Connection Issues**
```bash
# Restart Metabase service
docker-compose restart metabase

# Check ClickHouse from Metabase container
docker-compose exec metabase ping clickhouse
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats

# Check ClickHouse performance
docker-compose exec clickhouse clickhouse-client -q "SHOW PROCESSLIST"
```

## 📝 Development

### Adding New Data Sources
1. Update `etl_pipeline.py` with new extraction method
2. Modify ClickHouse schema in `init-scripts/`
3. Add validation rules in `data_validation.py`
4. Create corresponding Metabase visualizations

### Custom Metrics
1. Define calculation logic in transformation phase
2. Add to `daily_production_metrics` table schema
3. Create dashboard questions in `metabase_setup.py`

## 🔒 Security Considerations

- Change default passwords in production
- Use environment-specific configuration files
- Implement proper network security
- Regular security updates for all components

## 📋 Requirements Met

✅ **Data Pipeline Design**
- Extract from SQL, CSV, and API sources
- Transform into business metrics
- Load into ClickHouse data warehouse

✅ **ETL Implementation**
- Python script with comprehensive error handling
- SQL operations for database interactions
- Automated data processing workflows

✅ **Data Validation**
- Range and constraint validation
- Anomaly detection and logging
- Comprehensive quality scoring

✅ **Dashboard Creation**
- Metabase integration with ClickHouse
- Interactive visualizations
- Real-time data updates

✅ **Documentation & Version Control**
- Comprehensive technical documentation
- Git version control
- Reproducible deployment process

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in the `/logs` directory
3. Consult the technical report for detailed information

## 📜 License

This project is part of the Synapsis AI Engineer Technical Challenge.

---

**Last Updated**: December 2024  
**Version**: 1.0
