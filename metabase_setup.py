#!/usr/bin/env python3
"""
Metabase Setup and Dashboard Configuration Script
Author: Data Engineering Team
Description: Automated setup of Metabase dashboards for coal mining data visualization
"""

import requests
import json
import time
import logging
import os
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metabase_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MetabaseSetup:
    def __init__(self, metabase_url: str = "http://localhost:3000"):
        """Initialize Metabase setup with API connection"""
        self.base_url = metabase_url
        self.session_token = None
        self.database_id = None
        
        # ClickHouse connection details
        self.clickhouse_config = {
            "engine": "clickhouse",
            "name": "Coal Mining ClickHouse",
            "details": {
                "host": os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
                "port": int(os.getenv('CLICKHOUSE_PORT', 8123)),
                "dbname": os.getenv('CLICKHOUSE_DB', 'coal_mining'),
                "user": os.getenv('CLICKHOUSE_USER', 'default'),
                "password": os.getenv('CLICKHOUSE_PASSWORD', 'password'),
                "ssl": False,
                "tunnel-enabled": False
            }
        }

    def wait_for_metabase(self, max_attempts: int = 30, delay: int = 10):
        """Wait for Metabase to be ready"""
        logger.info("Waiting for Metabase to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Metabase is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            logger.info(f"Attempt {attempt + 1}/{max_attempts} - Waiting {delay} seconds...")
            time.sleep(delay)
        
        logger.error("Metabase failed to start within the expected time")
        return False

    def setup_initial_user(self) -> bool:
        """Setup initial admin user if not already done"""
        logger.info("Setting up initial admin user...")
        
        try:
            # Check if setup is already done
            setup_response = requests.get(f"{self.base_url}/api/session/properties")
            
            if setup_response.status_code == 200:
                properties = setup_response.json()
                if properties.get('setup-token'):
                    # Setup not yet completed
                    setup_data = {
                        "token": properties['setup-token'],
                        "user": {
                            "first_name": "Coal Mining",
                            "last_name": "Admin",
                            "email": "admin@coalmining.com",
                            "password": "CoalMining123!"
                        },
                        "database": None,
                        "invite": None,
                        "prefs": {
                            "site_name": "Coal Mining Analytics",
                            "allow_tracking": False
                        }
                    }
                    
                    setup_response = requests.post(
                        f"{self.base_url}/api/setup",
                        json=setup_data,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if setup_response.status_code == 200:
                        logger.info("Initial setup completed successfully")
                        return True
                    else:
                        logger.error(f"Setup failed: {setup_response.text}")
                        return False
                else:
                    logger.info("Metabase setup already completed")
                    return True
            
        except Exception as e:
            logger.error(f"Error during initial setup: {e}")
            return False

    def login(self) -> bool:
        """Login to Metabase and get session token"""
        logger.info("Logging into Metabase...")
        
        try:
            login_data = {
                "username": "admin@coalmining.com",
                "password": "CoalMining123!"
            }
            
            response = requests.post(
                f"{self.base_url}/api/session",
                json=login_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                self.session_token = response.json()['id']
                logger.info("Successfully logged into Metabase")
                return True
            else:
                logger.error(f"Login failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False

    def add_clickhouse_database(self) -> bool:
        """Add ClickHouse database connection to Metabase"""
        logger.info("Adding ClickHouse database connection...")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Metabase-Session": self.session_token
            }
            
            response = requests.post(
                f"{self.base_url}/api/database",
                json=self.clickhouse_config,
                headers=headers
            )
            
            if response.status_code == 200:
                self.database_id = response.json()['id']
                logger.info(f"ClickHouse database added successfully with ID: {self.database_id}")
                
                # Sync database schema
                sync_response = requests.post(
                    f"{self.base_url}/api/database/{self.database_id}/sync_schema",
                    headers=headers
                )
                
                if sync_response.status_code == 200:
                    logger.info("Database schema synchronized successfully")
                    return True
                else:
                    logger.warning("Database added but schema sync failed")
                    return True
                    
            else:
                logger.error(f"Failed to add database: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding ClickHouse database: {e}")
            return False

    def create_dashboard_questions(self) -> Dict[str, int]:
        """Create the questions (queries) needed for the dashboard"""
        logger.info("Creating dashboard questions...")
        
        questions = {}
        
        # Define the SQL queries for each visualization
        queries = {
            "Daily Production Trend": """
                SELECT 
                    date,
                    total_production_daily as "Total Production (tons)"
                FROM daily_production_metrics
                ORDER BY date
            """,
            
            "Average Quality Grade": """
                SELECT 
                    date,
                    average_quality_grade as "Quality Grade"
                FROM daily_production_metrics
                ORDER BY date
            """,
            
            "Equipment Utilization": """
                SELECT 
                    date,
                    equipment_utilization as "Utilization %"
                FROM daily_production_metrics
                ORDER BY date
            """,
            
            "Fuel Efficiency": """
                SELECT 
                    date,
                    fuel_efficiency as "Fuel per Ton (L)"
                FROM daily_production_metrics
                ORDER BY date
            """,
            
            "Weather Impact Analysis": """
                SELECT 
                    date,
                    precipitation_sum as "Precipitation (mm)",
                    temperature_mean as "Temperature (°C)",
                    weather_impact_score as "Weather Impact Score"
                FROM daily_production_metrics
                ORDER BY date
            """,
            
            "Production by Mine": """
                SELECT 
                    m.mine_name as "Mine",
                    SUM(p.tons_extracted) as "Total Production (tons)"
                FROM production_logs p
                JOIN mines m ON p.mine_id = m.mine_id
                GROUP BY m.mine_name
                ORDER BY "Total Production (tons)" DESC
            """,
            
            "Monthly Production Summary": """
                SELECT 
                    toStartOfMonth(date) as "Month",
                    SUM(total_production_daily) as "Total Production (tons)",
                    AVG(average_quality_grade) as "Average Quality",
                    AVG(equipment_utilization) as "Average Utilization %"
                FROM daily_production_metrics
                GROUP BY toStartOfMonth(date)
                ORDER BY "Month"
            """,
            
            "Equipment Status Distribution": """
                SELECT 
                    status as "Status",
                    COUNT(*) as "Count"
                FROM equipment_sensors
                GROUP BY status
                ORDER BY "Count" DESC
            """,
            
            "Production vs Weather Correlation": """
                SELECT 
                    precipitation_sum as "Precipitation (mm)",
                    total_production_daily as "Production (tons)"
                FROM daily_production_metrics
                WHERE precipitation_sum IS NOT NULL
                AND total_production_daily IS NOT NULL
            """
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Metabase-Session": self.session_token
        }
        
        for question_name, sql_query in queries.items():
            try:
                question_data = {
                    "name": question_name,
                    "dataset_query": {
                        "type": "native",
                        "native": {
                            "query": sql_query
                        },
                        "database": self.database_id
                    },
                    "display": self.get_display_type(question_name),
                    "visualization_settings": self.get_visualization_settings(question_name)
                }
                
                response = requests.post(
                    f"{self.base_url}/api/card",
                    json=question_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    question_id = response.json()['id']
                    questions[question_name] = question_id
                    logger.info(f"Created question: {question_name} (ID: {question_id})")
                else:
                    logger.error(f"Failed to create question '{question_name}': {response.text}")
                    
            except Exception as e:
                logger.error(f"Error creating question '{question_name}': {e}")
        
        return questions

    def get_display_type(self, question_name: str) -> str:
        """Get appropriate display type for each question"""
        if "Trend" in question_name or "Analysis" in question_name:
            return "line"
        elif "Summary" in question_name:
            return "table"
        elif "Distribution" in question_name or "by Mine" in question_name:
            return "pie"
        elif "Correlation" in question_name:
            return "scatter"
        else:
            return "line"

    def get_visualization_settings(self, question_name: str) -> Dict[str, Any]:
        """Get visualization settings for each question"""
        settings = {}
        
        if "Production" in question_name:
            settings["graph.colors"] = ["#31698D", "#86BCB6", "#F9E79F"]
        elif "Quality" in question_name:
            settings["graph.colors"] = ["#E74C3C", "#F39C12", "#2ECC71"]
        elif "Weather" in question_name:
            settings["graph.colors"] = ["#3498DB", "#9B59B6", "#1ABC9C"]
        
        return settings

    def create_dashboard(self, questions: Dict[str, int]) -> bool:
        """Create the main dashboard with all questions"""
        logger.info("Creating Coal Mining Analytics Dashboard...")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Metabase-Session": self.session_token
            }
            
            dashboard_data = {
                "name": "Coal Mining Production Analytics",
                "description": "Comprehensive analytics dashboard for coal mining operations including production metrics, equipment utilization, and weather impact analysis."
            }
            
            response = requests.post(
                f"{self.base_url}/api/dashboard",
                json=dashboard_data,
                headers=headers
            )
            
            if response.status_code == 200:
                dashboard_id = response.json()['id']
                logger.info(f"Dashboard created successfully (ID: {dashboard_id})")
                
                # Add questions to dashboard
                self.add_questions_to_dashboard(dashboard_id, questions)
                return True
                
            else:
                logger.error(f"Failed to create dashboard: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return False

    def add_questions_to_dashboard(self, dashboard_id: int, questions: Dict[str, int]):
        """Add questions to the dashboard with appropriate layout"""
        logger.info("Adding questions to dashboard...")
        
        headers = {
            "Content-Type": "application/json",
            "X-Metabase-Session": self.session_token
        }
        
        # Define layout positions for each question
        layout_positions = {
            "Daily Production Trend": {"col": 0, "row": 0, "sizeX": 8, "sizeY": 4},
            "Average Quality Grade": {"col": 8, "row": 0, "sizeX": 8, "sizeY": 4},
            "Equipment Utilization": {"col": 0, "row": 4, "sizeX": 6, "sizeY": 4},
            "Fuel Efficiency": {"col": 6, "row": 4, "sizeX": 6, "sizeY": 4},
            "Weather Impact Analysis": {"col": 12, "row": 4, "sizeX": 4, "sizeY": 4},
            "Production by Mine": {"col": 0, "row": 8, "sizeX": 6, "sizeY": 4},
            "Monthly Production Summary": {"col": 6, "row": 8, "sizeX": 10, "sizeY": 4},
            "Equipment Status Distribution": {"col": 0, "row": 12, "sizeX": 6, "sizeY": 4},
            "Production vs Weather Correlation": {"col": 6, "row": 12, "sizeX": 10, "sizeY": 4}
        }
        
        dashcards = []
        
        for question_name, question_id in questions.items():
            if question_name in layout_positions:
                position = layout_positions[question_name]
                
                dashcard_data = {
                    "cardId": question_id,
                    "col": position["col"],
                    "row": position["row"],
                    "sizeX": position["sizeX"],
                    "sizeY": position["sizeY"]
                }
                
                dashcards.append(dashcard_data)
        
        # Add all dashcards to dashboard
        try:
            response = requests.put(
                f"{self.base_url}/api/dashboard/{dashboard_id}/cards",
                json={"cards": dashcards},
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully added {len(dashcards)} visualizations to dashboard")
                return True
            else:
                logger.error(f"Failed to add questions to dashboard: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding questions to dashboard: {e}")
            return False

    def setup_complete_metabase(self) -> bool:
        """Run the complete Metabase setup process"""
        logger.info("Starting complete Metabase setup...")
        
        # Wait for Metabase to be ready
        if not self.wait_for_metabase():
            return False
        
        # Setup initial user
        if not self.setup_initial_user():
            return False
        
        # Login
        if not self.login():
            return False
        
        # Add ClickHouse database
        if not self.add_clickhouse_database():
            return False
        
        # Wait a bit for database sync
        time.sleep(10)
        
        # Create questions
        questions = self.create_dashboard_questions()
        if not questions:
            logger.error("No questions were created successfully")
            return False
        
        # Create dashboard
        if not self.create_dashboard(questions):
            return False
        
        logger.info("Metabase setup completed successfully!")
        logger.info(f"Access your dashboard at: {self.base_url}")
        logger.info("Login credentials:")
        logger.info("  Email: admin@coalmining.com")
        logger.info("  Password: CoalMining123!")
        
        return True

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run Metabase setup
    metabase_setup = MetabaseSetup()
    success = metabase_setup.setup_complete_metabase()
    
    if success:
        print("\n=== Metabase Setup Complete ===")
        print("Dashboard URL: http://localhost:3000")
        print("Username: admin@coalmining.com")
        print("Password: CoalMining123!")
    else:
        print("Metabase setup failed. Check logs for details.")
