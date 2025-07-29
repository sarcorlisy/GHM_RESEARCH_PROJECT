"""
Database Connector Module for Hospital Readmission Prediction Pipeline

This module provides database connectivity and operations for the analytics pipeline.
It supports PostgreSQL database operations and can be easily extended for other databases.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, Dict, List, Any
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Database connector class for managing PostgreSQL connections and operations.
    
    This class provides methods for:
    - Database connection management
    - Data insertion and retrieval
    - Table creation and management
    - Query execution and result processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database connector with configuration.
        
        Args:
            config: Database configuration dictionary containing:
                   - host: Database host
                   - port: Database port
                   - database: Database name
                   - user: Username
                   - password: Password
        """
        self.config = config or self._load_default_config()
        self.connection = None
        self.engine = None
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default database configuration."""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'hospital_readmission'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create SQLAlchemy engine for pandas operations
            connection_string = (
                f"postgresql://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def create_tables(self) -> bool:
        """
        Create necessary tables for the hospital readmission prediction pipeline.
        
        Returns:
            bool: True if tables created successfully, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Create patients table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS patients (
                        patient_id VARCHAR(50) PRIMARY KEY,
                        age INTEGER,
                        gender VARCHAR(10),
                        race VARCHAR(50),
                        admission_type_id INTEGER,
                        discharge_disposition_id INTEGER,
                        admission_source_id INTEGER,
                        time_in_hospital INTEGER,
                        num_lab_procedures INTEGER,
                        num_procedures INTEGER,
                        num_medications INTEGER,
                        number_outpatient INTEGER,
                        number_emergency INTEGER,
                        number_inpatient INTEGER,
                        diag_1 VARCHAR(10),
                        diag_2 VARCHAR(10),
                        diag_3 VARCHAR(10),
                        readmitted VARCHAR(10),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create encounters table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS encounters (
                        encounter_id VARCHAR(50) PRIMARY KEY,
                        patient_id VARCHAR(50) REFERENCES patients(patient_id),
                        encounter_date DATE,
                        encounter_type VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create medications table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS medications (
                        id SERIAL PRIMARY KEY,
                        patient_id VARCHAR(50) REFERENCES patients(patient_id),
                        medication_name VARCHAR(100),
                        dosage_change VARCHAR(10),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create model_results table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_results (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(50),
                        feature_selection_method VARCHAR(50),
                        top_n_features INTEGER,
                        accuracy DECIMAL(5,4),
                        precision DECIMAL(5,4),
                        recall DECIMAL(5,4),
                        f1_score DECIMAL(5,4),
                        auc_score DECIMAL(5,4),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("Database tables created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            return False
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                        if_exists: str = 'replace') -> bool:
        """
        Insert pandas DataFrame into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Data inserted into {table_name} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {str(e)}")
            return False
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            if params:
                df = pd.read_sql_query(query, self.engine, params=params)
            else:
                df = pd.read_sql_query(query, self.engine)
            
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            return pd.DataFrame()
    
    def get_patient_readmission_stats(self) -> pd.DataFrame:
        """
        Get readmission statistics by patient demographics.
        
        Returns:
            pd.DataFrame: Readmission statistics
        """
        query = """
        SELECT 
            age,
            gender,
            COUNT(*) as total_patients,
            SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as early_readmissions,
            SUM(CASE WHEN readmitted = '>30' THEN 1 ELSE 0 END) as late_readmissions,
            SUM(CASE WHEN readmitted = 'NO' THEN 1 ELSE 0 END) as no_readmissions,
            ROUND(
                SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) as early_readmission_rate
        FROM patients 
        GROUP BY age, gender 
        ORDER BY age, gender
        """
        return self.execute_query(query)
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get model performance summary from database.
        
        Returns:
            pd.DataFrame: Model performance data
        """
        query = """
        SELECT 
            model_name,
            feature_selection_method,
            top_n_features,
            AVG(accuracy) as avg_accuracy,
            AVG(precision) as avg_precision,
            AVG(recall) as avg_recall,
            AVG(f1_score) as avg_f1_score,
            AVG(auc_score) as avg_auc_score,
            COUNT(*) as num_runs
        FROM model_results 
        GROUP BY model_name, feature_selection_method, top_n_features
        ORDER BY avg_auc_score DESC
        """
        return self.execute_query(query)
    
    def save_model_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Save model training result to database.
        
        Args:
            result_data: Dictionary containing model result data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            query = """
            INSERT INTO model_results 
            (model_name, feature_selection_method, top_n_features, 
             accuracy, precision, recall, f1_score, auc_score)
            VALUES (%(model_name)s, %(feature_selection_method)s, %(top_n_features)s,
                    %(accuracy)s, %(precision)s, %(recall)s, %(f1_score)s, %(auc_score)s)
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(query), result_data)
                conn.commit()
            
            logger.info("Model result saved to database successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model result: {str(e)}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary from database.
        
        Returns:
            Dict: Summary statistics
        """
        try:
            summary = {}
            
            # Patient count
            patient_count = self.execute_query("SELECT COUNT(*) as count FROM patients")
            summary['total_patients'] = patient_count['count'].iloc[0] if not patient_count.empty else 0
            
            # Readmission distribution
            readmission_dist = self.execute_query("""
                SELECT readmitted, COUNT(*) as count 
                FROM patients 
                GROUP BY readmitted
            """)
            summary['readmission_distribution'] = readmission_dist.to_dict('records')
            
            # Age distribution
            age_stats = self.execute_query("""
                SELECT 
                    MIN(age) as min_age,
                    MAX(age) as max_age,
                    AVG(age) as avg_age,
                    STDDEV(age) as std_age
                FROM patients
            """)
            summary['age_statistics'] = age_stats.to_dict('records')[0] if not age_stats.empty else {}
            
            # Gender distribution
            gender_dist = self.execute_query("""
                SELECT gender, COUNT(*) as count 
                FROM patients 
                GROUP BY gender
            """)
            summary['gender_distribution'] = gender_dist.to_dict('records')
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {str(e)}")
            return {}


class DatabaseManager:
    """
    High-level database manager for the hospital readmission pipeline.
    
    This class provides convenient methods for common database operations
    specific to the hospital readmission prediction workflow.
    """
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize database manager.
        
        Args:
            db_config: Database configuration
        """
        self.connector = DatabaseConnector(db_config)
        
    def initialize_database(self) -> bool:
        """
        Initialize database with tables and basic setup.
        
        Returns:
            bool: True if initialization successful
        """
        if not self.connector.connect():
            return False
        
        if not self.connector.create_tables():
            return False
        
        logger.info("Database initialized successfully")
        return True
    
    def migrate_csv_to_database(self, csv_file_path: str, table_name: str) -> bool:
        """
        Migrate CSV data to database table.
        
        Args:
            csv_file_path: Path to CSV file
            table_name: Target table name
            
        Returns:
            bool: True if migration successful
        """
        try:
            df = pd.read_csv(csv_file_path)
            return self.connector.insert_dataframe(df, table_name)
            
        except Exception as e:
            logger.error(f"Failed to migrate CSV to database: {str(e)}")
            return False
    
    def get_analytics_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get analytics data for dashboard and reporting.
        
        Returns:
            Dict: Dictionary containing various analytics DataFrames
        """
        analytics_data = {
            'readmission_stats': self.connector.get_patient_readmission_stats(),
            'model_performance': self.connector.get_model_performance_summary(),
            'data_summary': self.connector.get_data_summary()
        }
        
        return analytics_data
    
    def close(self):
        """Close database connection."""
        self.connector.disconnect()


# Example usage and testing
if __name__ == "__main__":
    # Test database connection
    db_manager = DatabaseManager()
    
    if db_manager.initialize_database():
        print("Database initialized successfully")
        
        # Example: Get data summary
        summary = db_manager.get_analytics_data()
        print("Data summary:", json.dumps(summary['data_summary'], indent=2))
        
        db_manager.close()
    else:
        print("Failed to initialize database") 