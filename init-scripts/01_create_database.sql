-- Create database
CREATE DATABASE IF NOT EXISTS coal_mining;

-- Switch to the database
USE coal_mining;

-- Create mines table
CREATE TABLE IF NOT EXISTS mines (
    mine_id UInt32,
    mine_code String,
    mine_name String,
    location String,
    operational_status String
) ENGINE = MergeTree()
ORDER BY mine_id;

-- Create production_logs table  
CREATE TABLE IF NOT EXISTS production_logs (
    log_id UInt32,
    date Date,
    mine_id UInt32,
    shift String,
    tons_extracted Decimal(10,2),
    quality_grade Decimal(3,1)
) ENGINE = MergeTree()
ORDER BY (date, mine_id, shift);

-- Create equipment_sensors table
CREATE TABLE IF NOT EXISTS equipment_sensors (
    timestamp DateTime,
    equipment_id String,
    status String,
    fuel_consumption Float64,
    maintenance_alert Bool
) ENGINE = MergeTree()
ORDER BY (timestamp, equipment_id);

-- Create weather_data table
CREATE TABLE IF NOT EXISTS weather_data (
    date Date,
    latitude Float64,
    longitude Float64,
    temperature_mean Float64,
    precipitation_sum Float64,
    timezone String
) ENGINE = MergeTree()
ORDER BY date;

-- Create daily_production_metrics table (target table for ETL)
CREATE TABLE IF NOT EXISTS daily_production_metrics (
    date Date,
    total_production_daily Float64,
    average_quality_grade Float64,
    equipment_utilization Float64,
    fuel_efficiency Float64,
    weather_impact_score Float64,
    temperature_mean Float64,
    precipitation_sum Float64,
    anomaly_flags Array(String),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY date;

-- Create data_quality_log table for anomaly tracking
CREATE TABLE IF NOT EXISTS data_quality_log (
    id UInt64,
    timestamp DateTime DEFAULT now(),
    table_name String,
    anomaly_type String,
    description String,
    affected_records UInt32,
    severity String
) ENGINE = MergeTree()
ORDER BY (timestamp, table_name);
