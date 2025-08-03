"""
MySQL Database Connector

Specialized connector and query executor designed for MySQL database
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import logging

class MySQLConnector:
    """MySQL Database Connector"""
    
    def __init__(self, config=None):
        """
        Initialize MySQL connector
        
        Args:
            config (dict): Database configuration
        """
        self.config = config or self._get_default_config()
        self.connection = None
        self.engine = None
        self._setup_logging()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'database': os.getenv('DB_NAME', 'hospital_readmission'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'charset': 'utf8mb4',
            'autocommit': True
        }
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to MySQL database"""
        try:
            # Create database connection
            self.connection = mysql.connector.connect(**self.config)
            
            # Create SQLAlchemy engine
            connection_string = (
                f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string)
            
            self.logger.info(f"✅ Successfully connected to MySQL database: {self.config['host']}:{self.config['port']}")
            return True
            
        except Error as e:
            self.logger.error(f"❌ MySQL connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def create_database_if_not_exists(self):
        """Create database (if it doesn't exist)"""
        try:
            # Temporary connection (without specifying database)
            temp_config = self.config.copy()
            temp_config.pop('database', None)
            
            temp_connection = mysql.connector.connect(**temp_config)
            cursor = temp_connection.cursor()
            
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            self.logger.info(f"✅ Database {self.config['database']} created or already exists")
            
            cursor.close()
            temp_connection.close()
            
            return True
            
        except Error as e:
            self.logger.error(f"❌ Failed to create database: {e}")
            return False
    
    def create_tables(self):
        """Create necessary tables"""
        try:
            cursor = self.connection.cursor()
            
            # Create patients table
            create_patients_table = """
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(50),
                encounter_id VARCHAR(50),
                race VARCHAR(50),
                gender VARCHAR(50),
                age VARCHAR(50),
                weight VARCHAR(50),
                admission_type_id INT,
                discharge_disposition_id INT,
                admission_source_id INT,
                time_in_hospital INT,
                payer_code VARCHAR(50),
                medical_specialty VARCHAR(200),
                num_lab_procedures INT,
                num_procedures INT,
                num_medications INT,
                number_outpatient INT,
                number_emergency INT,
                number_inpatient INT,
                diag_1 VARCHAR(50),
                diag_2 VARCHAR(50),
                diag_3 VARCHAR(50),
                number_diagnoses INT,
                max_glu_serum VARCHAR(50),
                A1Cresult VARCHAR(50),
                metformin VARCHAR(50),
                repaglinide VARCHAR(50),
                nateglinide VARCHAR(50),
                chlorpropamide VARCHAR(50),
                glimepiride VARCHAR(50),
                acetohexamide VARCHAR(50),
                glipizide VARCHAR(50),
                glyburide VARCHAR(50),
                tolbutamide VARCHAR(50),
                pioglitazone VARCHAR(50),
                rosiglitazone VARCHAR(50),
                acarbose VARCHAR(50),
                miglitol VARCHAR(50),
                troglitazone VARCHAR(50),
                tolazamide VARCHAR(50),
                examide VARCHAR(50),
                citoglipton VARCHAR(50),
                insulin VARCHAR(50),
                glyburide_metformin VARCHAR(50),
                glipizide_metformin VARCHAR(50),
                glimepiride_pioglitazone VARCHAR(50),
                metformin_rosiglitazone VARCHAR(50),
                metformin_pioglitazone VARCHAR(50),
                medication_change VARCHAR(50),
                diabetesMed VARCHAR(50),
                readmitted VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            cursor.execute(create_patients_table)
            self.logger.info("✅ Patients table created successfully")
            
            cursor.close()
            return True
            
        except Error as e:
            self.logger.error(f"❌ Failed to create tables: {e}")
            return False
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """
        Insert DataFrame into database table
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: What to do if table exists ('fail', 'replace', 'append')
        """
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            self.logger.info(f"✅ Successfully inserted {len(df)} rows into {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to insert DataFrame: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """
        Execute SELECT query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            self.logger.error(f"❌ Query execution failed: {e}")
            return None
    
    def execute_update(self, query, params=None):
        """
        Execute UPDATE/INSERT/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            affected_rows = cursor.rowcount
            self.connection.commit()
            cursor.close()
            self.logger.info(f"✅ Query executed successfully, {affected_rows} rows affected")
            return affected_rows
        except Error as e:
            self.logger.error(f"❌ Update execution failed: {e}")
            return 0
    
    def get_table_info(self, table_name):
        """
        Get table structure information
        
        Args:
            table_name: Table name
            
        Returns:
            Table structure information
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            cursor.close()
            return columns
        except Error as e:
            self.logger.error(f"❌ Failed to get table info: {e}")
            return None
    
    def get_table_count(self, table_name):
        """
        Get row count of table
        
        Args:
            table_name: Table name
            
        Returns:
            Row count
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Error as e:
            self.logger.error(f"❌ Failed to get table count: {e}")
            return 0


class MySQLManager:
    """MySQL Database Manager - High-level operations"""
    
    def __init__(self, config=None):
        """Initialize MySQL manager"""
        self.connector = MySQLConnector(config)
    
    def initialize_database(self):
        """Initialize database and tables"""
        try:
            # Create database if not exists
            if not self.connector.create_database_if_not_exists():
                return False
            
            # Connect to database
            if not self.connector.connect():
                return False
            
            # Create tables
            if not self.connector.create_tables():
                return False
            
            self.logger.info("✅ Database initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def migrate_csv_to_database(self, csv_file, table_name, chunk_size=1000):
        """
        Migrate CSV file to database table
        
        Args:
            csv_file: CSV file path
            table_name: Target table name
            chunk_size: Number of rows to process at once
        """
        try:
            # Read CSV in chunks
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                self.connector.insert_dataframe(chunk, table_name, if_exists='append')
            
            self.logger.info(f"✅ CSV migration to {table_name} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CSV migration failed: {e}")
            return False
    
    def get_database_summary(self):
        """Get database summary information"""
        try:
            tables = ['patients']
            summary = {}
            
            for table in tables:
                count = self.connector.get_table_count(table)
                summary[table] = count
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get database summary: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        self.connector.disconnect() 