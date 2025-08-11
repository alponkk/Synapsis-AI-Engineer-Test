-- Create extension for additional data types
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create mines table
CREATE TABLE IF NOT EXISTS mines (
    mine_id SERIAL PRIMARY KEY,
    mine_code VARCHAR(10) NOT NULL,
    mine_name VARCHAR(50) NOT NULL,
    location VARCHAR(100) NOT NULL,
    operational_status VARCHAR(20) NOT NULL
);

-- Create production_logs table  
CREATE TABLE IF NOT EXISTS production_logs (
    log_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    mine_id INTEGER NOT NULL,
    shift VARCHAR(10) NOT NULL,
    tons_extracted DECIMAL(10,2),
    quality_grade DECIMAL(3,1),
    FOREIGN KEY (mine_id) REFERENCES mines(mine_id)
);

-- Create equipment_sensors table
CREATE TABLE IF NOT EXISTS equipment_sensors (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    equipment_id VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    fuel_consumption DOUBLE PRECISION,
    maintenance_alert BOOLEAN
);

-- Create weather_data table
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    temperature_mean DOUBLE PRECISION,
    precipitation_sum DOUBLE PRECISION,
    timezone VARCHAR(50)
);

-- Create daily_production_metrics table (target table for ETL)
CREATE TABLE IF NOT EXISTS daily_production_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    total_production_daily DOUBLE PRECISION,
    average_quality_grade DOUBLE PRECISION,
    equipment_utilization DOUBLE PRECISION,
    fuel_efficiency DOUBLE PRECISION,
    weather_impact_score DOUBLE PRECISION,
    temperature_mean DOUBLE PRECISION,
    precipitation_sum DOUBLE PRECISION,
    anomaly_flags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create data_quality_log table for anomaly tracking
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    table_name VARCHAR(50),
    anomaly_type VARCHAR(50),
    description TEXT,
    affected_records INTEGER,
    severity VARCHAR(20)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_production_logs_date ON production_logs(date);
CREATE INDEX IF NOT EXISTS idx_production_logs_mine_id ON production_logs(mine_id);
CREATE INDEX IF NOT EXISTS idx_equipment_sensors_timestamp ON equipment_sensors(timestamp);
CREATE INDEX IF NOT EXISTS idx_equipment_sensors_equipment_id ON equipment_sensors(equipment_id);
CREATE INDEX IF NOT EXISTS idx_weather_data_date ON weather_data(date);
CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_production_metrics(date);
CREATE INDEX IF NOT EXISTS idx_quality_log_timestamp ON data_quality_log(timestamp);
