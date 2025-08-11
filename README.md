# Coal Mining Data Pipeline

A comprehensive data engineering solution for optimizing coal mining operations through automated data collection, transformation, and visualization. This pipeline integrates production data from SQL databases, IoT sensor data from CSV files, and weather data from external APIs into a unified PostgreSQL data warehouse.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed and running
- 8GB+ RAM recommended
- Ports 5432 available (PostgreSQL)

### One-Command Deployment
```bash
# Clone the repository
git clone <repository-url>

# Start all services
docker-compose up -d --build

# Wait for initialization (30-60 seconds)
```

## 📊 System Overview

### Technology Stack
- **Database**: PostgreSQL 15 (relational database for analytics)
- **ETL Framework**: Python 3.11 with pandas, psycopg2, SQLAlchemy
- **Containerization**: Docker Compose
- **Visualization**: Metabase v0.47.7
- **APIs**: Open-Meteo Weather API
- **Version Control**: Git

### Data Sources
1. **Production Logs**: SQL database containing daily mining operations data
2. **Equipment Sensors**: CSV files with hourly IoT sensor readings
3. **Weather API**: Real-time weather data for Berau, Kalimantan, Indonesia

### Key Metrics Generated
- **Total Production Daily**: Daily coal extraction totals (full load - replaces all data)
- **Average Quality Grade**: Coal quality assessment
- **Equipment Utilization**: Operational efficiency percentages
- **Fuel Efficiency**: Fuel consumption per ton of coal
- **Weather Impact**: Correlation between weather and production

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Sources│───▶│ ETL Pipeline│───▶│ PostgreSQL  │───▶│  Metabase   │
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
├── python_scripts/                # Python scripts directory
│   ├── etl_pipeline.py            # Main ETL script
│   └── data_validation.py         # Data quality validation
├── init-scripts/                  # Database initialization
│   └── 01_create_database.sql
├── synapsis ai engineer challege datasets/
│   ├── production_logs.sql        # Production data
│   └── equipment_sensors.csv      # Sensor data
├── logs/                          # Application logs
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables
```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=coal_mining
```

### Service Ports
- **PostgreSQL**: 5432
- **Metabase**: 3000

### Metabase Access
**Email**: admin@coalmining.com
**Password**: CoalMining123!
**First Name**: Coal
**Last Name**: Mining


## 🛠️ Operations

### Manual ETL Execution
The ETL pipeline now runs only when manually executed (no automatic scheduling):

```bash
# Run ETL pipeline manually
docker-compose exec etl_runner python python_scripts/etl_pipeline.py

# Run data validation manually
docker-compose exec etl_runner python python_scripts/data_validation.py
```

### Accessing Metabase Dashboard
```bash
# Access Metabase at http://localhost:3000
# Note: Metabase has been manually configured
# Connect to PostgreSQL database within Metabase using:
# Host: postgres
# Port: 5432
# Database: coal_mining
# Username: postgres
# Password: password
```

### Viewing Logs
```bash
# ETL runner logs
docker-compose logs etl_runner

# PostgreSQL logs
docker-compose logs postgres

# Metabase logs
docker-compose logs metabase

# All service logs
docker-compose logs -f
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d coal_mining

# Example queries
SELECT * FROM daily_production_metrics LIMIT 10;
SELECT COUNT(*) FROM production_logs;
```

## 📈 Data Pipeline Details

### ETL Process Implementation

#### 1. Extraction Phase

**Production Data Extraction**
- **Source**: `production_logs.sql` file containing SQL INSERT statements
- **Process**: Regex parsing to extract structured data from SQL statements
- **Tables**: `mines` and `production_logs`
- **Volume**: 2,000+ production records across multiple mines and shifts

**Sensor Data Extraction**
- **Source**: `equipment_sensors.csv` with hourly equipment readings
- **Process**: Pandas CSV reading with data type conversion
- **Fields**: timestamp, equipment_id, status, fuel_consumption, maintenance_alert
- **Volume**: 40,000+ sensor readings

**Weather Data Extraction**
- **Source**: Open-Meteo API (https://api.open-meteo.com/v1/forecast)
- **Location**: Berau, Kalimantan (2.0167°N, 117.3000°E)
- **Parameters**: Daily temperature and precipitation
- **Error Handling**: Fallback to default values for API failures

#### 2. Transformation Phase

**Data Cleaning and Validation**
- **Negative Values**: Replace negative `tons_extracted` with 0
- **Quality Grades**: Cap values to valid range (0-10)
- **Missing Data**: Forward-fill equipment utilization data
- **Anomaly Flagging**: Track and log all data quality issues

**Metric Generation**
The pipeline generates five key business metrics:

1. **Total Production Daily**
   ```sql
   SUM(tons_extracted) GROUP BY date
   ```

2. **Average Quality Grade**
   ```sql
   AVG(quality_grade) GROUP BY date
   ```

3. **Equipment Utilization**
   ```sql
   (COUNT(status = 'active') / COUNT(*)) * 100 GROUP BY date, equipment_id
   ```

4. **Fuel Efficiency**
   ```sql
   AVG(fuel_consumption) / SUM(tons_extracted) GROUP BY date
   ```

5. **Weather Impact Score**
   - Algorithm: `100 - (precipitation_impact + temperature_impact)`
   - Precipitation Impact: `(precipitation - 5) * 2` for precipitation > 5mm
   - Temperature Impact: `ABS(temperature - 25) * 1.5`

#### 3. Loading Phase (Full Load Strategy)
- **Target**: PostgreSQL tables
- **Strategy**: Full load - truncates existing data and replaces with new data
- **Performance**: Batch loading with SQLAlchemy for optimal throughput
- **Logging**: Comprehensive audit trail of all operations

**Full Load Implementation:**
- `mines`: Uses `if_exists='replace'`
- `production_logs`: Uses `if_exists='replace'`
- `equipment_sensors`: Uses `if_exists='replace'`
- `weather_data`: Uses `if_exists='replace'`
- `daily_production_metrics`: Truncates table before inserting new data

## 🔍 Data Validation and Quality Assurance

### Validation Framework
The system implements a multi-layered validation approach:

#### Schema Validation
- **Structure Checks**: Verify table existence and required columns
- **Data Types**: Ensure proper data type constraints
- **Referential Integrity**: Validate foreign key relationships

#### Data Quality Validation
- **Range Checks**: Validate numeric values within expected ranges
- **Null Checks**: Monitor and report missing data
- **Duplicate Detection**: Identify and flag duplicate records
- **Format Validation**: Ensure date and string formats are correct

#### Business Rule Validation
- **Production Rules**: `total_production_daily >= 0`
- **Utilization Rules**: `0 <= equipment_utilization <= 100`
- **Quality Rules**: `0 <= quality_grade <= 10`
- **Efficiency Rules**: `fuel_efficiency > 0`

### Anomaly Detection
The system employs statistical methods for anomaly detection:

#### Statistical Analysis
- **IQR Method**: Detect outliers using interquartile range
- **Z-Score Analysis**: Identify values beyond 3 standard deviations
- **Trend Analysis**: Monitor production trends over time

#### Alert System
- **Severity Levels**: High, Medium, Low based on statistical significance
- **Automated Logging**: All anomalies logged to `data_quality_log` table
- **Recommendations**: Automated suggestions for data quality improvement

### Data Quality Scoring
The system calculates a comprehensive quality score (0-100) based on:
- Null values (max 20 point deduction)
- Out-of-range values (max 30 point deduction)
- Duplicates (max 15 point deduction)
- Anomalies (max 25 point deduction)

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

### Equipment Performance
```sql
-- Equipment status distribution
SELECT 
    status,
    COUNT(*) as count,
    AVG(fuel_consumption) as avg_fuel
FROM equipment_sensors
GROUP BY status;
```

## 🚨 Troubleshooting

### Common Issues

**ETL Pipeline Fails**
```bash
# Check data source files
ls -la "synapsis ai engineer challege datasets/"

# Verify PostgreSQL connectivity
docker-compose exec etl_runner python -c "import psycopg2; print('PostgreSQL connection OK')"

# Check ETL logs
docker-compose logs etl_runner
```

**PostgreSQL Connection Issues**
```bash
# Restart PostgreSQL service
docker-compose restart postgres

# Check PostgreSQL status
docker-compose exec postgres pg_isready -U postgres
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats

# Check PostgreSQL performance
docker-compose exec postgres psql -U postgres -d coal_mining -c "SELECT * FROM pg_stat_activity;"
```

## 📝 Development

### Adding New Data Sources
1. Update `python_scripts/etl_pipeline.py` with new extraction method
2. Modify PostgreSQL schema in `init-scripts/`
3. Add validation rules in `python_scripts/data_validation.py`

### Custom Metrics
1. Define calculation logic in transformation phase
2. Add to `daily_production_metrics` table schema
3. Test with data validation scripts

## 🔒 Security Considerations

- Change default passwords in production
- Use environment-specific configuration files
- Implement proper network security
- Regular security updates for all components
- Secure API keys and database credentials

## 📋 Key Features

✅ **Data Pipeline Design**
- Extract from SQL, CSV, and API sources
- Transform into business metrics
- Load into PostgreSQL data warehouse with full load strategy

✅ **ETL Implementation**
- Python script with comprehensive error handling
- SQL operations for database interactions
- Manual execution workflow

✅ **Data Validation**
- Range and constraint validation
- Anomaly detection and logging
- Comprehensive quality scoring

✅ **Data Visualization**
- Metabase dashboard integration
- Interactive coal mining analytics
- Real-time data visualization
- Manual Metabase configuration (no automated setup required)

✅ **Full Load Strategy**
- Complete data refresh on each run
- Ensures data consistency
- Truncate and reload approach

✅ **Documentation & Version Control**
- Comprehensive technical documentation
- Git version control
- Reproducible deployment process

This project is part of the Synapsis AI Engineer Technical Challenge.