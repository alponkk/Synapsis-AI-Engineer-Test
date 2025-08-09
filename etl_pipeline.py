#!/usr/bin/env python3
"""
Coal Mining Data Pipeline ETL Script
Author: Data Engineering Team
Description: Extract, Transform, Load coal mining production data from multiple sources
"""

import os
import pandas as pd
import numpy as np
import clickhouse_connect
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
        """Initialize the ETL pipeline with ClickHouse connection and configuration"""
        self.ch_host = os.getenv('CLICKHOUSE_HOST', 'localhost')
        self.ch_port = int(os.getenv('CLICKHOUSE_PORT', 8123))
        self.ch_user = os.getenv('CLICKHOUSE_USER', 'default')
        self.ch_password = os.getenv('CLICKHOUSE_PASSWORD', 'password')
        self.ch_database = os.getenv('CLICKHOUSE_DB', 'coal_mining')
        
        # Weather API configuration
        self.weather_api_url = "https://api.open-meteo.com/v1/forecast"
        self.latitude = 2.0167
        self.longitude = 117.3000
        
        self.client = None
        self.connect_to_clickhouse()

    def connect_to_clickhouse(self):
        """Establish connection to ClickHouse"""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.ch_host,
                port=self.ch_port,
                username=self.ch_user,
                password=self.ch_password,
                database=self.ch_database
            )
            logger.info("Successfully connected to ClickHouse")
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    def extract_production_data(self) -> pd.DataFrame:
        """Extract production data from SQL file and load to ClickHouse"""
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
            
            # Load mines data to ClickHouse
            if mines_data:
                mines_df = pd.DataFrame(mines_data)
                self.client.insert('mines', mines_df.values.tolist(), column_names=mines_df.columns.tolist())
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
            
            # Load production data to ClickHouse
            if production_data:
                production_df = pd.DataFrame(production_data)
                # Add log_id as auto-increment
                production_df['log_id'] = range(1, len(production_df) + 1)
                # Reorder columns
                production_df = production_df[['log_id', 'date', 'mine_id', 'shift', 'tons_extracted', 'quality_grade']]
                
                self.client.insert('production_logs', production_df.values.tolist(), column_names=production_df.columns.tolist())
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
            
            # Load to ClickHouse
            self.client.insert('equipment_sensors', sensor_df.values.tolist(), column_names=sensor_df.columns.tolist())
            logger.info(f"Loaded {len(sensor_df)} sensor records")
            
            return sensor_df
            
        except Exception as e:
            logger.error(f"Error extracting sensor data: {e}")
            raise

    def extract_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract weather data from Open-Meteo API"""
        logger.info(f"Extracting weather data from {start_date} to {end_date}")
        
        weather_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                params = {
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                    'daily': 'temperature_2m_mean,precipitation_sum',
                    'timezone': 'Asia/Jakarta',
                    'start_date': date_str,
                    'end_date': date_str
                }
                
                response = requests.get(self.weather_api_url, params=params, timeout=10)
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
                    
            except Exception as e:
                logger.warning(f"Failed to fetch weather data for {date_str}: {e}")
                # Use default values for missing data
                weather_data.append({
                    'date': date_str,
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                    'temperature_mean': 26.0,  # Default temperature
                    'precipitation_sum': 0.0,  # Default no rain
                    'timezone': 'Asia/Jakarta'
                })
            
            current_date += timedelta(days=1)
        
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
            
            # Load to ClickHouse
            self.client.insert('weather_data', weather_df.values.tolist(), column_names=weather_df.columns.tolist())
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
            # Get data from ClickHouse
            production_query = """
                SELECT date, mine_id, shift, tons_extracted, quality_grade
                FROM production_logs
                ORDER BY date, mine_id, shift
            """
            production_df = self.client.query_df(production_query)
            
            sensor_query = """
                SELECT 
                    toDate(timestamp) as date,
                    equipment_id,
                    status,
                    fuel_consumption,
                    maintenance_alert
                FROM equipment_sensors
                ORDER BY timestamp
            """
            sensor_df = self.client.query_df(sensor_query)
            
            weather_query = """
                SELECT date, temperature_mean, precipitation_sum
                FROM weather_data
                ORDER BY date
            """
            weather_df = self.client.query_df(weather_query)
            
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
        """Load the transformed data into ClickHouse"""
        logger.info("Loading data into daily_production_metrics table")
        
        try:
            # Validate metrics before loading
            validation_errors = self.validate_metrics(df)
            
            # Add validation errors to anomaly flags if any
            if validation_errors:
                for i, row in df.iterrows():
                    df.at[i, 'anomaly_flags'] = df.at[i, 'anomaly_flags'] + validation_errors
            
            # Prepare data for ClickHouse
            # Convert date column to string for ClickHouse compatibility
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
            
            # Clear existing data for the date range
            min_date = df_final['date'].min()
            max_date = df_final['date'].max()
            
            delete_query = f"""
                ALTER TABLE daily_production_metrics 
                DELETE WHERE date >= '{min_date}' AND date <= '{max_date}'
            """
            self.client.command(delete_query)
            
            # Insert new data
            self.client.insert('daily_production_metrics', df_final.values.tolist(), column_names=df_final.columns.tolist())
            
            logger.info(f"Successfully loaded {len(df_final)} records into daily_production_metrics")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def log_data_quality_issue(self, table_name: str, anomaly_type: str, description: str, affected_records: int, severity: str):
        """Log data quality issues to the data_quality_log table"""
        try:
            # Get next ID
            max_id_query = "SELECT max(id) as max_id FROM data_quality_log"
            result = self.client.query(max_id_query)
            max_id = result.result_rows[0][0] if result.result_rows and result.result_rows[0][0] is not None else 0
            next_id = max_id + 1
            
            log_data = [{
                'id': next_id,
                'table_name': table_name,
                'anomaly_type': anomaly_type,
                'description': description,
                'affected_records': affected_records,
                'severity': severity
            }]
            
            self.client.insert('data_quality_log', [list(log_data[0].values())], column_names=list(log_data[0].keys()))
            
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
