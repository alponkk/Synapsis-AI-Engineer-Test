#!/usr/bin/env python3
"""
Data Validation and Quality Monitoring Script
Author: Data Engineering Team
Description: Comprehensive data validation and anomaly detection for coal mining data
"""

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        """Initialize the data validator with PostgreSQL connection"""
        self.pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.pg_port = int(os.getenv('POSTGRES_PORT', 5432))
        self.pg_user = os.getenv('POSTGRES_USER', 'postgres')
        self.pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        self.pg_database = os.getenv('POSTGRES_DB', 'coal_mining')
        
        self.engine = None
        self.connect_to_postgres()
        
        # Validation rules and thresholds
        self.validation_rules = {
            'production_logs': {
                'tons_extracted': {'min': 0, 'max': 1000, 'type': float},
                'quality_grade': {'min': 0, 'max': 10, 'type': float},
                'required_fields': ['date', 'mine_id', 'shift', 'tons_extracted', 'quality_grade']
            },
            'equipment_sensors': {
                'fuel_consumption': {'min': 0, 'max': 50, 'type': float},
                'status': {'allowed_values': ['active', 'idle', 'maintenance'], 'type': str},
                'required_fields': ['timestamp', 'equipment_id', 'status', 'fuel_consumption', 'maintenance_alert']
            },
            'weather_data': {
                'temperature_mean': {'min': -10, 'max': 50, 'type': float},
                'precipitation_sum': {'min': 0, 'max': 500, 'type': float},
                'required_fields': ['date', 'latitude', 'longitude', 'temperature_mean', 'precipitation_sum']
            },
            'daily_production_metrics': {
                'total_production_daily': {'min': 0, 'max': 10000, 'type': float},
                'average_quality_grade': {'min': 0, 'max': 10, 'type': float},
                'equipment_utilization': {'min': 0, 'max': 100, 'type': float},
                'fuel_efficiency': {'min': 0, 'max': 20, 'type': float},
                'weather_impact_score': {'min': 0, 'max': 100, 'type': float},
                'required_fields': ['date', 'total_production_daily', 'average_quality_grade', 
                                  'equipment_utilization', 'fuel_efficiency', 'weather_impact_score']
            }
        }

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
            
            logger.info("Successfully connected to PostgreSQL for validation")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def validate_table_structure(self, table_name: str) -> Dict[str, Any]:
        """Validate table structure and required fields"""
        logger.info(f"Validating structure for table: {table_name}")
        
        validation_result = {
            'table_name': table_name,
            'structure_valid': True,
            'missing_fields': [],
            'extra_fields': [],
            'total_records': 0,
            'errors': []
        }
        
        try:
            # Get table structure (PostgreSQL syntax)
            structure_query = f"""
                SELECT column_name as name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                AND table_schema = 'public'
            """
            structure = pd.read_sql(structure_query, self.engine)
            
            # Get column names
            existing_columns = structure['name'].tolist()
            
            # Check required fields
            if table_name in self.validation_rules:
                required_fields = self.validation_rules[table_name]['required_fields']
                missing_fields = [field for field in required_fields if field not in existing_columns]
                
                if missing_fields:
                    validation_result['structure_valid'] = False
                    validation_result['missing_fields'] = missing_fields
                    logger.error(f"Missing required fields in {table_name}: {missing_fields}")
            
            # Get record count
            count_query = f"SELECT COUNT(*) as total FROM {table_name}"
            count_result = pd.read_sql(count_query, self.engine)
            validation_result['total_records'] = int(count_result['total'].iloc[0])
            
            logger.info(f"Table {table_name} has {validation_result['total_records']} records")
            
        except Exception as e:
            validation_result['structure_valid'] = False
            validation_result['errors'].append(str(e))
            logger.error(f"Error validating table structure for {table_name}: {e}")
        
        return validation_result

    def validate_data_quality(self, table_name: str, date_filter: str = None) -> Dict[str, Any]:
        """Validate data quality for a specific table"""
        logger.info(f"Validating data quality for table: {table_name}")
        
        validation_result = {
            'table_name': table_name,
            'validation_date': datetime.now().isoformat(),
            'records_checked': 0,
            'anomalies': [],
            'null_counts': {},
            'out_of_range_counts': {},
            'duplicate_counts': 0,
            'quality_score': 100.0,
            'recommendations': []
        }
        
        try:
            # Build query with optional date filter
            base_query = f"SELECT * FROM {table_name}"
            if date_filter:
                if table_name in ['production_logs', 'weather_data', 'daily_production_metrics']:
                    base_query += f" WHERE date >= '{date_filter}'"
                elif table_name == 'equipment_sensors':
                    base_query += f" WHERE DATE(timestamp) >= '{date_filter}'"
            
            # Get data
            df = pd.read_sql(base_query, self.engine)
            validation_result['records_checked'] = len(df)
            
            if len(df) == 0:
                logger.warning(f"No data found in {table_name}")
                return validation_result
            
            # Check for null values
            null_counts = df.isnull().sum().to_dict()
            validation_result['null_counts'] = {k: int(v) for k, v in null_counts.items() if v > 0}
            
            # Check data ranges and constraints
            if table_name in self.validation_rules:
                rules = self.validation_rules[table_name]
                
                for column, constraints in rules.items():
                    if column == 'required_fields':
                        continue
                        
                    if column in df.columns:
                        if 'min' in constraints and 'max' in constraints:
                            out_of_range = ((df[column] < constraints['min']) | 
                                          (df[column] > constraints['max'])).sum()
                            if out_of_range > 0:
                                validation_result['out_of_range_counts'][column] = int(out_of_range)
                                validation_result['anomalies'].append({
                                    'type': 'out_of_range',
                                    'column': column,
                                    'count': int(out_of_range),
                                    'expected_range': f"{constraints['min']}-{constraints['max']}"
                                })
                        
                        if 'allowed_values' in constraints:
                            invalid_values = ~df[column].isin(constraints['allowed_values'])
                            invalid_count = invalid_values.sum()
                            if invalid_count > 0:
                                validation_result['anomalies'].append({
                                    'type': 'invalid_values',
                                    'column': column,
                                    'count': int(invalid_count),
                                    'allowed_values': constraints['allowed_values']
                                })
            
            # Check for duplicates
            if table_name == 'production_logs':
                duplicate_mask = df.duplicated(subset=['date', 'mine_id', 'shift'])
            elif table_name == 'equipment_sensors':
                duplicate_mask = df.duplicated(subset=['timestamp', 'equipment_id'])
            elif table_name == 'weather_data':
                duplicate_mask = df.duplicated(subset=['date'])
            elif table_name == 'daily_production_metrics':
                duplicate_mask = df.duplicated(subset=['date'])
            else:
                duplicate_mask = df.duplicated()
            
            validation_result['duplicate_counts'] = int(duplicate_mask.sum())
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(validation_result, len(df))
            validation_result['quality_score'] = quality_score
            
            # Generate recommendations
            recommendations = self.generate_recommendations(validation_result)
            validation_result['recommendations'] = recommendations
            
            logger.info(f"Data quality validation completed for {table_name}. Quality score: {quality_score:.1f}")
            
        except Exception as e:
            logger.error(f"Error during data quality validation for {table_name}: {e}")
            validation_result['errors'] = [str(e)]
        
        return validation_result

    def calculate_quality_score(self, validation_result: Dict[str, Any], total_records: int) -> float:
        """Calculate a data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for null values
        null_penalty = sum(validation_result['null_counts'].values()) / total_records * 100
        score -= min(null_penalty * 0.5, 20)  # Max 20 points deduction for nulls
        
        # Deduct points for out-of-range values
        range_penalty = sum(validation_result['out_of_range_counts'].values()) / total_records * 100
        score -= min(range_penalty * 0.8, 30)  # Max 30 points deduction for range issues
        
        # Deduct points for duplicates
        duplicate_penalty = validation_result['duplicate_counts'] / total_records * 100
        score -= min(duplicate_penalty * 0.3, 15)  # Max 15 points deduction for duplicates
        
        # Deduct points for anomalies
        anomaly_penalty = len(validation_result['anomalies']) * 5  # 5 points per anomaly type
        score -= min(anomaly_penalty, 25)  # Max 25 points deduction for anomalies
        
        return max(score, 0.0)

    def generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if validation_result['null_counts']:
            recommendations.append("Implement data completeness checks to reduce null values")
        
        if validation_result['out_of_range_counts']:
            recommendations.append("Add data range validation at ingestion to prevent out-of-range values")
        
        if validation_result['duplicate_counts'] > 0:
            recommendations.append("Implement deduplication logic to prevent duplicate records")
        
        if validation_result['quality_score'] < 90:
            recommendations.append("Overall data quality needs improvement - review data collection processes")
        
        if validation_result['quality_score'] < 70:
            recommendations.append("Critical data quality issues detected - immediate attention required")
        
        return recommendations

    def detect_production_anomalies(self, days_to_check: int = 30) -> Dict[str, Any]:
        """Detect anomalies in production data using statistical methods"""
        logger.info(f"Detecting production anomalies for the last {days_to_check} days")
        
        anomaly_result = {
            'analysis_date': datetime.now().isoformat(),
            'days_analyzed': days_to_check,
            'anomalies_detected': [],
            'statistical_summary': {},
            'alerts': []
        }
        
        try:
            # Get recent production data
            query = f"""
                SELECT 
                    date,
                    total_production_daily,
                    average_quality_grade,
                    equipment_utilization,
                    fuel_efficiency,
                    weather_impact_score
                FROM daily_production_metrics
                WHERE date >= CURRENT_DATE - INTERVAL '{days_to_check} days'
                ORDER BY date
            """
            
            df = pd.read_sql(query, self.engine)
            
            if len(df) < 7:  # Need at least a week of data
                logger.warning("Insufficient data for anomaly detection")
                return anomaly_result
            
            # Statistical analysis for each metric
            metrics = ['total_production_daily', 'average_quality_grade', 'equipment_utilization', 
                      'fuel_efficiency', 'weather_impact_score']
            
            for metric in metrics:
                if metric in df.columns:
                    series = df[metric]
                    
                    # Calculate statistics
                    mean_val = series.mean()
                    std_val = series.std()
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    
                    anomaly_result['statistical_summary'][metric] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'q1': float(q1),
                        'q3': float(q3),
                        'min': float(series.min()),
                        'max': float(series.max())
                    }
                    
                    # Detect outliers using IQR method
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = df[(series < lower_bound) | (series > upper_bound)]
                    
                    if len(outliers) > 0:
                        for _, row in outliers.iterrows():
                            anomaly_result['anomalies_detected'].append({
                                'date': str(row['date']),
                                'metric': metric,
                                'value': float(row[metric]),
                                'expected_range': f"{lower_bound:.2f} - {upper_bound:.2f}",
                                'severity': 'high' if abs(row[metric] - mean_val) > 3 * std_val else 'medium'
                            })
                    
                    # Generate alerts for critical thresholds
                    if metric == 'total_production_daily' and series.iloc[-1] < mean_val - 2 * std_val:
                        anomaly_result['alerts'].append(f"Production significantly below average: {series.iloc[-1]:.2f} tons")
                    
                    if metric == 'equipment_utilization' and series.iloc[-1] < 50:
                        anomaly_result['alerts'].append(f"Low equipment utilization: {series.iloc[-1]:.1f}%")
                    
                    if metric == 'average_quality_grade' and series.iloc[-1] < 3.0:
                        anomaly_result['alerts'].append(f"Poor coal quality detected: {series.iloc[-1]:.1f}")
            
            # Trend analysis
            production_trend = self.analyze_trend(df['total_production_daily'])
            if production_trend == 'declining':
                anomaly_result['alerts'].append("Declining production trend detected over the analysis period")
            
            logger.info(f"Anomaly detection completed. Found {len(anomaly_result['anomalies_detected'])} anomalies")
            
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            anomaly_result['errors'] = [str(e)]
        
        return anomaly_result

    def analyze_trend(self, series: pd.Series, threshold: float = 0.1) -> str:
        """Analyze trend in a time series"""
        if len(series) < 5:
            return 'insufficient_data'
        
        # Simple linear trend analysis
        x = np.arange(len(series))
        coefficients = np.polyfit(x, series, 1)
        slope = coefficients[0]
        
        # Normalize slope by mean to get percentage change per period
        normalized_slope = slope / series.mean() if series.mean() != 0 else 0
        
        if normalized_slope > threshold:
            return 'increasing'
        elif normalized_slope < -threshold:
            return 'declining'
        else:
            return 'stable'

    def check_weather_data_completeness(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Check if weather data is complete for the given date range"""
        logger.info(f"Checking weather data completeness from {start_date} to {end_date}")
        
        completeness_result = {
            'start_date': start_date,
            'end_date': end_date,
            'expected_days': 0,
            'actual_days': 0,
            'missing_dates': [],
            'completeness_percentage': 0.0,
            'data_complete': False
        }
        
        try:
            # Calculate expected number of days
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            expected_days = (end_dt - start_dt).days + 1
            completeness_result['expected_days'] = expected_days
            
            # Get actual weather data
            query = f"""
                SELECT DISTINCT date
                FROM weather_data
                WHERE date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
            """
            
            weather_dates = pd.read_sql(query, self.engine)
            actual_days = len(weather_dates)
            completeness_result['actual_days'] = actual_days
            
            # Find missing dates
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            existing_dates = pd.to_datetime(weather_dates['date'])
            missing_dates = set(all_dates) - set(existing_dates)
            
            completeness_result['missing_dates'] = [d.strftime('%Y-%m-%d') for d in sorted(missing_dates)]
            completeness_result['completeness_percentage'] = (actual_days / expected_days) * 100 if expected_days > 0 else 0
            completeness_result['data_complete'] = len(missing_dates) == 0
            
            logger.info(f"Weather data completeness: {completeness_result['completeness_percentage']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error checking weather data completeness: {e}")
            completeness_result['errors'] = [str(e)]
        
        return completeness_result

    def generate_validation_report(self, output_file: str = 'logs/validation_report.json'):
        """Generate a comprehensive validation report"""
        logger.info("Generating comprehensive validation report")
        
        report = {
            'report_date': datetime.now().isoformat(),
            'validation_summary': {},
            'table_validations': {},
            'anomaly_analysis': {},
            'weather_completeness': {},
            'overall_assessment': {},
            'recommendations': []
        }
        
        try:
            # Validate all tables
            tables = ['production_logs', 'equipment_sensors', 'weather_data', 'daily_production_metrics']
            
            for table in tables:
                logger.info(f"Running validation for {table}")
                
                # Structure validation
                structure_result = self.validate_table_structure(table)
                
                # Data quality validation
                quality_result = self.validate_data_quality(table, date_filter='2024-07-01')
                
                report['table_validations'][table] = {
                    'structure': structure_result,
                    'quality': quality_result
                }
            
            # Anomaly detection
            report['anomaly_analysis'] = self.detect_production_anomalies(30)
            
            # Weather data completeness
            report['weather_completeness'] = self.check_weather_data_completeness('2024-07-01', '2024-12-31')
            
            # Overall assessment
            quality_scores = [result['quality']['quality_score'] for result in report['table_validations'].values()]
            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            report['overall_assessment'] = {
                'average_quality_score': avg_quality_score,
                'data_health': 'excellent' if avg_quality_score >= 95 else 
                             'good' if avg_quality_score >= 85 else 
                             'fair' if avg_quality_score >= 70 else 'poor',
                'critical_issues': len([s for s in quality_scores if s < 70]),
                'total_anomalies': len(report['anomaly_analysis'].get('anomalies_detected', [])),
                'weather_data_complete': report['weather_completeness'].get('data_complete', False)
            }
            
            # Generate overall recommendations
            if avg_quality_score < 85:
                report['recommendations'].append("Implement automated data quality monitoring")
            
            if report['overall_assessment']['total_anomalies'] > 5:
                report['recommendations'].append("Investigate root causes of production anomalies")
            
            if not report['weather_completeness'].get('data_complete', False):
                report['recommendations'].append("Ensure weather data collection is reliable and complete")
            
            # Save report
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {output_file}")
            logger.info(f"Overall data health: {report['overall_assessment']['data_health']}")
            logger.info(f"Average quality score: {avg_quality_score:.1f}")
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            report['errors'] = [str(e)]
        
        return report

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run data validation
    validator = DataValidator()
    report = validator.generate_validation_report()
    
    # Print summary
    print("\n=== Data Validation Summary ===")
    print(f"Overall Data Health: {report['overall_assessment']['data_health']}")
    print(f"Average Quality Score: {report['overall_assessment']['average_quality_score']:.1f}")
    print(f"Critical Issues: {report['overall_assessment']['critical_issues']}")
    print(f"Total Anomalies: {report['overall_assessment']['total_anomalies']}")
    print(f"Weather Data Complete: {report['overall_assessment']['weather_data_complete']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
