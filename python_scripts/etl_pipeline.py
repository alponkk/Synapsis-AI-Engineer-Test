#!/usr/bin/env python3
"""
Coal Mining Data Pipeline ETL Script
Author: Data Engineering Team
Description: Extract, Transform, Load coal mining production data from multiple sources
"""

import os
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
import requests
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CoalMiningETL:
    def __init__(self):
        """Initialize the ETL pipeline with PostgreSQL connection and configuration"""
        self.pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.pg_port = int(os.getenv('POSTGRES_PORT', 5432))
        self.pg_user = os.getenv('POSTGRES_USER', 'postgres')
        self.pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        self.pg_database = os.getenv('POSTGRES_DB', 'coal_mining')
        
        # Weather API configuration
        self.weather_forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.weather_historical_url = "https://api.open-meteo.com/v1/historical-weather"
        self.latitude = 2.0167
        self.longitude = 117.3000
        
        self.engine = None
        self.connection = None
        self.connect_to_postgres()

    def connect_to_postgres(self):
        """Establish connection to PostgreSQL"""
        try:
            # Create SQLAlchemy engine
            connection_string = f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info("Successfully connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def extract_production_data(self) -> pd.DataFrame:
        """Extract production data from SQL file and load to PostgreSQL"""
        logger.info("Extracting production data from SQL file")
        
        try:
            # Read the SQL file
            sql_file_path = "synapsis ai engineer challege datasets/production_logs.sql"
            with open(sql_file_path, 'r') as file:
                sql_content = file.read()
            
            # Parse INSERT statements for mines
            mines_data = []
            mine_pattern = r"INSERT INTO mines \([^)]+\) VALUES \(([^)]+)\);"
            mine_matches = re.findall(mine_pattern, sql_content)
            
            for match in mine_matches:
                values = [v.strip().strip("'") for v in match.split(',')]
                mines_data.append({
                    'mine_id': int(values[0]),
                    'mine_code': values[1],
                    'mine_name': values[2],
                    'location': values[3],
                    'operational_status': values[4]
                })
            
            # Load mines data to PostgreSQL (truncate first due to foreign key constraints)
            if mines_data:
                mines_df = pd.DataFrame(mines_data)
                
                # Truncate tables with foreign key relationships in proper order
                with self.engine.connect() as conn:
                    # First truncate dependent tables
                    conn.execute(text("TRUNCATE TABLE production_logs CASCADE"))
                    # Then truncate the referenced table
                    conn.execute(text("TRUNCATE TABLE mines CASCADE"))
                    conn.commit()
                
                mines_df.to_sql('mines', self.engine, if_exists='append', index=False, method='multi')
                logger.info(f"Loaded {len(mines_data)} mine records")
            
            # Parse INSERT statements for production logs
            production_data = []
            prod_pattern = r"INSERT INTO production_logs \([^)]+\) VALUES \(([^)]+)\);"
            prod_matches = re.findall(prod_pattern, sql_content)
            
            for match in prod_matches:
                values = [v.strip().strip("'") for v in match.split(',')]
                production_data.append({
                    'date': values[0],
                    'mine_id': int(values[1]),
                    'shift': values[2],
                    'tons_extracted': float(values[3]),
                    'quality_grade': float(values[4])
                })
            
            # Load production data to PostgreSQL
            if production_data:
                production_df = pd.DataFrame(production_data)
                # PostgreSQL will auto-generate log_id with SERIAL, so we don't include it
                production_df = production_df[['date', 'mine_id', 'shift', 'tons_extracted', 'quality_grade']]
                
                production_df.to_sql('production_logs', self.engine, if_exists='append', index=False, method='multi')
                logger.info(f"Loaded {len(production_data)} production log records")
                
                return production_df
            
        except Exception as e:
            logger.error(f"Error extracting production data: {e}")
            raise

    def extract_sensor_data(self) -> pd.DataFrame:
        """Extract equipment sensor data from CSV file"""
        logger.info("Extracting sensor data from CSV file")
        
        try:
            csv_file_path = "synapsis ai engineer challege datasets/equipment_sensors.csv"
            sensor_df = pd.read_csv(csv_file_path)
            
            # Convert timestamp to datetime
            sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
            
            # Convert maintenance_alert to boolean
            sensor_df['maintenance_alert'] = sensor_df['maintenance_alert'].astype(bool)
            
            # Load to PostgreSQL (truncate first for full load)
            with self.engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE equipment_sensors"))
                conn.commit()
            
            sensor_df.to_sql('equipment_sensors', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Loaded {len(sensor_df)} sensor records")
            
            return sensor_df
            
        except Exception as e:
            logger.error(f"Error extracting sensor data: {e}")
            raise

    def extract_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract weather data from Open-Meteo API (historical or forecast)"""
        logger.info(f"Extracting weather data from {start_date} to {end_date}")
        
        weather_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.now().date()
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            current_date_obj = current_date.date()
            
            try:
                # Determine which API endpoint to use based on date
                if current_date_obj <= today:
                    # Use historical API for past dates
                    api_url = self.weather_historical_url
                    params = {
                        'latitude': self.latitude,
                        'longitude': self.longitude,
                        'daily': 'temperature_2m_mean,precipitation_sum',
                        'timezone': 'Asia/Jakarta',
                        'start_date': date_str,
                        'end_date': date_str
                    }
                else:
                    # Use forecast API for future dates
                    api_url = self.weather_forecast_url
                    params = {
                        'latitude': self.latitude,
                        'longitude': self.longitude,
                        'daily': 'temperature_2m_mean,precipitation_sum',
                        'timezone': 'Asia/Jakarta',
                        'start_date': date_str,
                        'end_date': date_str
                    }
                
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if 'daily' in data and data['daily']['time']:
                    weather_data.append({
                        'date': data['daily']['time'][0],
                        'latitude': data['latitude'],
                        'longitude': data['longitude'],
                        'temperature_mean': data['daily']['temperature_2m_mean'][0],
                        'precipitation_sum': data['daily']['precipitation_sum'][0],
                        'timezone': data['timezone']
                    })
                    logger.debug(f"Successfully fetched weather data for {date_str}")
                else:
                    raise ValueError("No weather data returned from API")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch weather data for {date_str}: {e}")
                # Use default values for missing data
                weather_data.append({
                    'date': date_str,
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                    'temperature_mean': 26.0,  # Default temperature for Kalimantan
                    'precipitation_sum': 0.0,  # Default no rain
                    'timezone': 'Asia/Jakarta'
                })
            
            current_date += timedelta(days=1)
        
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
            
            # Load to PostgreSQL (truncate first for full load)
            with self.engine.connect() as conn:
                conn.execute(text("TRUNCATE TABLE weather_data"))
                conn.commit()
            
            weather_df.to_sql('weather_data', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Loaded {len(weather_df)} weather records")
            
            return weather_df
        
        return pd.DataFrame()

    def validate_and_clean_data(self, production_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean production data, return cleaned data and anomaly flags"""
        logger.info("Validating and cleaning production data")
        
        anomalies = []
        
        # Check for negative tons_extracted
        negative_mask = production_df['tons_extracted'] < 0
        if negative_mask.any():
            negative_count = negative_mask.sum()
            anomalies.append(f"negative_tons_extracted_{negative_count}")
            logger.warning(f"Found {negative_count} records with negative tons_extracted")
            
            # Replace negative values with 0
            production_df.loc[negative_mask, 'tons_extracted'] = 0
            
            # Log to data quality table
            self.log_data_quality_issue(
                table_name="production_logs",
                anomaly_type="negative_tons_extracted",
                description=f"Found {negative_count} records with negative tons_extracted, replaced with 0",
                affected_records=negative_count,
                severity="WARNING"
            )
        
        # Check for unrealistic quality grades (should be between 0-10)
        quality_mask = (production_df['quality_grade'] < 0) | (production_df['quality_grade'] > 10)
        if quality_mask.any():
            quality_count = quality_mask.sum()
            anomalies.append(f"invalid_quality_grade_{quality_count}")
            logger.warning(f"Found {quality_count} records with invalid quality grades")
            
            # Cap quality grades to valid range
            production_df.loc[production_df['quality_grade'] < 0, 'quality_grade'] = 0
            production_df.loc[production_df['quality_grade'] > 10, 'quality_grade'] = 10
        
        return production_df, anomalies

    def transform_data(self) -> pd.DataFrame:
        """Transform data to generate required metrics"""
        logger.info("Transforming data to generate metrics")
        
        try:
            # Get data from PostgreSQL
            production_query = """
                SELECT date, mine_id, shift, tons_extracted, quality_grade
                FROM production_logs
                ORDER BY date, mine_id, shift
            """
            production_df = pd.read_sql(production_query, self.engine)
            
            sensor_query = """
                SELECT 
                    DATE(timestamp) as date,
                    equipment_id,
                    status,
                    fuel_consumption,
                    maintenance_alert
                FROM equipment_sensors
                ORDER BY timestamp
            """
            sensor_df = pd.read_sql(sensor_query, self.engine)
            
            weather_query = """
                SELECT date, temperature_mean, precipitation_sum
                FROM weather_data
                ORDER BY date
            """
            weather_df = pd.read_sql(weather_query, self.engine)
            
            # Validate and clean production data
            production_df, anomaly_flags = self.validate_and_clean_data(production_df)
            
            # Calculate daily production metrics
            daily_production = production_df.groupby('date').agg({
                'tons_extracted': 'sum',
                'quality_grade': 'mean'
            }).reset_index()
            
            daily_production.columns = ['date', 'total_production_daily', 'average_quality_grade']
            
            # Calculate equipment utilization per day
            equipment_daily = sensor_df.groupby(['date', 'equipment_id']).agg({
                'status': lambda x: (x == 'active').mean() * 100,  # Percentage active
                'fuel_consumption': 'mean'
            }).reset_index()
            
            # Aggregate equipment metrics by date
            equipment_metrics = equipment_daily.groupby('date').agg({
                'status': 'mean',  # Average utilization across all equipment
                'fuel_consumption': 'mean'
            }).reset_index()
            
            equipment_metrics.columns = ['date', 'equipment_utilization', 'avg_fuel_consumption']
            
            # Calculate fuel efficiency (fuel per ton of coal)
            # Merge production and equipment data
            daily_metrics = daily_production.merge(equipment_metrics, on='date', how='left')
            
            # Calculate fuel efficiency: total fuel consumption / total production
            daily_metrics['fuel_efficiency'] = np.where(
                daily_metrics['total_production_daily'] > 0,
                daily_metrics['avg_fuel_consumption'] / daily_metrics['total_production_daily'],
                0
            )
            
            # Merge with weather data
            daily_metrics = daily_metrics.merge(weather_df, on='date', how='left')
            
            # Calculate weather impact score
            daily_metrics['weather_impact_score'] = self.calculate_weather_impact(daily_metrics)
            
            # Handle missing values
            daily_metrics = self.handle_missing_values(daily_metrics)
            
            # Add anomaly flags
            daily_metrics['anomaly_flags'] = [anomaly_flags] * len(daily_metrics)
            
            logger.info(f"Generated metrics for {len(daily_metrics)} days")
            return daily_metrics
            
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise

    def calculate_weather_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weather impact score based on precipitation and temperature"""
        # Simple weather impact calculation:
        # - High precipitation (>5mm) reduces production efficiency
        # - Extreme temperatures (>30°C or <20°C) reduce efficiency
        # Score: 0-100 (100 = no weather impact, 0 = severe impact)
        
        impact_score = pd.Series(100.0, index=df.index)  # Start with no impact
        
        # Precipitation impact
        high_precip_mask = df['precipitation_sum'] > 5
        impact_score.loc[high_precip_mask] -= (df.loc[high_precip_mask, 'precipitation_sum'] - 5) * 2
        
        # Temperature impact
        temp_impact = np.abs(df['temperature_mean'] - 25)  # Optimal temperature: 25°C
        impact_score -= temp_impact * 1.5
        
        # Ensure score is between 0 and 100
        impact_score = np.clip(impact_score, 0, 100)
        
        return impact_score

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing equipment utilization with previous day's average
        df['equipment_utilization'] = df['equipment_utilization'].fillna(method='ffill').fillna(50.0)
        
        # Fill missing fuel consumption with previous day's average
        df['avg_fuel_consumption'] = df['avg_fuel_consumption'].fillna(method='ffill').fillna(5.0)
        
        # Fill missing weather data with defaults
        df['temperature_mean'] = df['temperature_mean'].fillna(26.0)
        df['precipitation_sum'] = df['precipitation_sum'].fillna(0.0)
        
        return df

    def validate_metrics(self, df: pd.DataFrame) -> List[str]:
        """Validate the generated metrics"""
        logger.info("Validating generated metrics")
        
        validation_errors = []
        
        # Check total_production_daily is not negative
        negative_production = df['total_production_daily'] < 0
        if negative_production.any():
            count = negative_production.sum()
            validation_errors.append(f"negative_total_production_{count}")
            logger.error(f"Found {count} days with negative total production")
        
        # Check equipment_utilization is between 0 and 100
        invalid_utilization = (df['equipment_utilization'] < 0) | (df['equipment_utilization'] > 100)
        if invalid_utilization.any():
            count = invalid_utilization.sum()
            validation_errors.append(f"invalid_equipment_utilization_{count}")
            logger.error(f"Found {count} days with invalid equipment utilization")
        
        # Check for missing weather data
        missing_weather = df[['temperature_mean', 'precipitation_sum']].isnull().any(axis=1)
        if missing_weather.any():
            count = missing_weather.sum()
            validation_errors.append(f"missing_weather_data_{count}")
            logger.warning(f"Found {count} days with missing weather data")
        
        return validation_errors

    def load_data(self, df: pd.DataFrame):
        """Load the transformed data into PostgreSQL"""
        logger.info("Loading data into daily_production_metrics table")
        
        try:
            # Validate metrics before loading
            validation_errors = self.validate_metrics(df)
            
            # Add validation errors to anomaly flags if any
            if validation_errors:
                for i, row in df.iterrows():
                    df.at[i, 'anomaly_flags'] = df.at[i, 'anomaly_flags'] + validation_errors
            
            # Prepare data for PostgreSQL
            # Convert date column to string for PostgreSQL compatibility
            df_to_load = df.copy()
            df_to_load['date'] = df_to_load['date'].astype(str)
            
            # Ensure all required columns are present
            required_columns = [
                'date', 'total_production_daily', 'average_quality_grade',
                'equipment_utilization', 'fuel_efficiency', 'weather_impact_score',
                'temperature_mean', 'precipitation_sum', 'anomaly_flags'
            ]
            
            # Select and reorder columns
            df_final = df_to_load[required_columns].copy()
            
            # Full load: Truncate all existing data first (PostgreSQL syntax)
            with self.engine.connect() as conn:
                truncate_query = text("TRUNCATE TABLE daily_production_metrics")
                conn.execute(truncate_query)
                conn.commit()
            
            # Insert new data
            df_final.to_sql('daily_production_metrics', self.engine, if_exists='append', index=False, method='multi')
            
            logger.info(f"Successfully loaded {len(df_final)} records into daily_production_metrics")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def log_data_quality_issue(self, table_name: str, anomaly_type: str, description: str, affected_records: int, severity: str):
        """Log data quality issues to the data_quality_log table"""
        try:
            log_data = pd.DataFrame([{
                'table_name': table_name,
                'anomaly_type': anomaly_type,
                'description': description,
                'affected_records': affected_records,
                'severity': severity
            }])
            
            log_data.to_sql('data_quality_log', self.engine, if_exists='append', index=False, method='multi')
            
        except Exception as e:
            logger.error(f"Error logging data quality issue: {e}")

    def run_etl_pipeline(self):
        """Run the complete ETL pipeline"""
        logger.info("Starting Coal Mining ETL Pipeline")
        
        try:
            # Step 1: Extract data
            logger.info("Step 1: Extracting data from all sources")
            production_df = self.extract_production_data()
            sensor_df = self.extract_sensor_data()
            
            # Get date range from production data
            min_date = production_df['date'].min()
            max_date = production_df['date'].max()
            
            weather_df = self.extract_weather_data(min_date, max_date)
            
            # Step 2: Transform data
            logger.info("Step 2: Transforming data and generating metrics")
            metrics_df = self.transform_data()
            
            # Step 3: Load data
            logger.info("Step 3: Loading data into data warehouse")
            self.load_data(metrics_df)
            
            logger.info("ETL Pipeline completed successfully!")
            
            # Print summary
            self.print_pipeline_summary(metrics_df)
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            raise

    def print_pipeline_summary(self, df: pd.DataFrame):
        """Print a summary of the pipeline results"""
        logger.info("=== Pipeline Summary ===")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Total days processed: {len(df)}")
        logger.info(f"Total coal production: {df['total_production_daily'].sum():.2f} tons")
        logger.info(f"Average daily production: {df['total_production_daily'].mean():.2f} tons")
        logger.info(f"Average quality grade: {df['average_quality_grade'].mean():.2f}")
        logger.info(f"Average equipment utilization: {df['equipment_utilization'].mean():.1f}%")
        logger.info(f"Average fuel efficiency: {df['fuel_efficiency'].mean():.2f} L/ton")
        logger.info(f"Average weather impact score: {df['weather_impact_score'].mean():.1f}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run ETL pipeline
    etl = CoalMiningETL()
    etl.run_etl_pipeline()