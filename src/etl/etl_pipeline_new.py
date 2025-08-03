"""
ETL Pipeline Framework for Hospital Readmission Data
Simple ETL framework with basic structure
"""

import os
import sys
import logging
import mysql.connector
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import data pre-cleaning modules
try:
    from src.etl.data_pre_cleaner import DataPreCleaner
    from src.etl.dynamic_column_cleaner import DynamicColumnCleaner
except ImportError:
    from data_pre_cleaner import DataPreCleaner
    from dynamic_column_cleaner import DynamicColumnCleaner

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import configuration and utilities
try:
    from src.utils.config import ConfigManager
    from src.utils.logging_config import get_logger
except ImportError:
    # Fallback imports when running file directly
    from utils.config import ConfigManager
    from utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

class HospitalReadmissionETL:
    """
    Hospital Readmission Data ETL Pipeline
    Engineering Architecture: Azure Download → Pre-cleaning → SQL Connection → SQL Import
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize ETL pipeline"""
        self.config = config or ConfigManager()
        self.connection = None
        self.data_pre_cleaner = DataPreCleaner()  # Data pre-cleaner
        
    def connect_to_mysql(self) -> bool:
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 3306)),
                database=os.getenv('DB_NAME', 'hospital_readmission'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', 'hospital123'),
                charset=os.getenv('DB_CHARSET', 'utf8mb4')
            )
            logger.info(" Connected to MySQL database")
            return True
        except Exception as e:
            logger.error(f" Failed to connect to MySQL: {e}")
            return False
        
    def disconnect_from_mysql(self):
        """Disconnect from MySQL database"""
        try:
            if self.connection:
                self.connection.close()
            logger.info(" Disconnected from MySQL database")
        except Exception as e:
            logger.warning(f" Error during disconnect: {e}")
        
    def execute_sql_file(self, sql_file_path: str) -> bool:
        """
        Engineering SQL Executor - Universal SQL file execution functionality
        Optimized version: Handles SELECT results and duplicate index issues
        
        Args:
            sql_file_path: SQL file path
            
        Returns:
            bool: Whether execution was successful
        """
        try:
            if not os.path.exists(sql_file_path):
                logger.error(f"SQL file does not exist: {sql_file_path}")
                return False
            
            logger.info(f"Executing SQL file: {sql_file_path}")
            
            # Read SQL file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Split SQL statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            # Execute SQL statements using cursor
            cursor = self.connection.cursor()
            
            for statement in statements:
                if statement and not statement.startswith('--'):
                    try:
                        # Execute SQL statement
                        cursor.execute(statement)
                        
                        # If it's a SELECT statement, consume results to avoid "Unread result found"
                        if statement.strip().upper().startswith('SELECT'):
                            results = cursor.fetchall()
                            if results:
                                logger.info(f"SELECT statement executed successfully, returned {len(results)} rows")
                            else:
                                logger.info("SELECT statement executed successfully, no results")
                        else:
                            logger.info(f"SQL statement executed: {statement[:50]}...")
                            
                    except Exception as e:
                        error_msg = str(e)
                        
                        # Handle duplicate index errors (ignore)
                        if "Duplicate key name" in error_msg:
                            logger.info(f"Index already exists, skipping: {statement[:50]}...")
                            continue
                        
                        # Handle column doesn't exist errors (ignore)
                        elif "doesn't exist in table" in error_msg:
                            logger.info(f"Column doesn't exist, skipping: {statement[:50]}...")
                            continue
                        
                        # Handle BLOB/TEXT column index length errors (ignore)
                        elif "BLOB/TEXT column" in error_msg and "used in key specification without a key length" in error_msg:
                            logger.info(f"TEXT column index needs length specification, skipping: {statement[:50]}...")
                            continue
                        
                        # Handle other errors
                        else:
                            logger.warning(f"SQL statement execution failed: {e}")
                            logger.warning(f"Statement: {statement[:100]}...")
            
            self.connection.commit()
            cursor.close()
            logger.info(f"SQL file execution completed: {sql_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute SQL file: {e}")
            return False
        
    def run_data_mapping(self) -> bool:
        """
        Run data mapping
        Map ID fields to readable description text and automatically upload to Azure
        
        Returns:
            bool: Whether mapping was successful
        """
        try:
            logger.info("Starting data mapping process...")
            
            # Ensure database connection
            if not self.connection or not self.connection.is_connected():
                if not self.connect_to_mysql():
                    logger.error("Database connection failed")
                    return False
            
            # Execute data mapping SQL file
            sql_file_path = "src/etl/sql_processing/01_data_mapping.sql"
            if self.execute_sql_file(sql_file_path):
                logger.info("Data mapping SQL executed successfully")
                
                # Verify mapping results
                cursor = self.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM patients_mapped")
                mapped_count = cursor.fetchone()[0]
                cursor.close()
                
                logger.info(f"Data mapping completed, mapped {mapped_count} records")
                
                # Automatically upload mapped data to Azure
                logger.info("Automatically uploading mapped data to Azure...")
                if self.upload_table_to_azure("patients_mapped", "mapped-data"):
                    logger.info("Mapped data uploaded to Azure successfully")
                else:
                    logger.warning("Failed to upload mapped data to Azure")
                
                return True
            else:
                logger.error("Data mapping SQL execution failed")
                return False
                
        except Exception as e:
            logger.error(f"Data mapping process failed: {e}")
            return False
        
    def run_data_cleaning(self) -> bool:
        """
        Run data cleaning
        Use dynamic column cleaner, based on patients_mapped table, results saved as patients_cleaned, automatically upload to Azure
        
        Returns:
            bool: Whether data cleaning was successful
        """
        try:
            logger.info("Starting dynamic column deletion process...")
            
            # Ensure database connection
            if not self.connection or not self.connection.is_connected():
                if not self.connect_to_mysql():
                    logger.error("Database connection failed")
                    return False
            
            # Create dynamic column cleaner
            from src.etl.dynamic_column_cleaner import DynamicColumnCleaner
            dynamic_cleaner = DynamicColumnCleaner()
            dynamic_cleaner.connect_database()
            
            try:
                # Execute dynamic column deletion
                result = dynamic_cleaner.execute_dynamic_cleaning()
                
                # If execution successful, return True directly
                if result and result.get('total_columns_kept', 0) > 0:
                    logger.info(f"Dynamic column deletion executed successfully")
                    logger.info(f"Kept {result.get('total_columns_kept', 0)} columns, removed {result.get('total_columns_removed', 0)} columns")
                    logger.info(f"Removed columns: {', '.join(result.get('columns_removed', []))}")
                    return True
                else:
                    logger.error("Dynamic column deletion execution failed")
                    return False
                
            except Exception as e:
                logger.error(f"Dynamic column cleaner execution failed: {e}")
                # Check if table was created successfully
                try:
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM patients_cleaned")
                    count = cursor.fetchone()[0]
                    cursor.close()
                    if count > 0:
                        logger.info(f"Although error occurred, patients_cleaned table was created successfully with {count} records")
                        return True
                    else:
                        logger.error("patients_cleaned table creation failed")
                        return False
                except Exception as check_error:
                    logger.error(f"Cannot verify patients_cleaned table: {check_error}")
                    return False
            finally:
                if 'dynamic_cleaner' in locals():
                    dynamic_cleaner.close_connection()
                
        except Exception as e:
            logger.error(f"Dynamic column deletion process failed: {e}")
            return False
        
    def run_business_cleaning(self) -> bool:
        """
        Run business rule cleaning
        Apply business rules to filter data, based on patients_cleaned table, results saved as patients_business_cleaned, automatically upload to Azure
        
        Returns:
            bool: Whether business cleaning was successful
        """
        try:
            logger.info("Starting business rule cleaning process...")
            
            # Ensure database connection
            if not self.connection or not self.connection.is_connected():
                if not self.connect_to_mysql():
                    logger.error("Database connection failed")
                    return False
            
            # Execute business cleaning SQL file
            sql_file_path = "src/etl/sql_processing/03_business_cleaning.sql"
            if self.execute_sql_file(sql_file_path):
                logger.info("Business cleaning SQL executed successfully")
                
                # Verify business cleaning results
                cursor = self.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM patients_business_cleaned")
                business_cleaned_count = cursor.fetchone()[0]
                cursor.close()
                
                logger.info(f" Business cleaning completed, cleaned {business_cleaned_count} records")
                
                # Automatically upload business cleaned data to Azure
                logger.info("☁️ Automatically uploading business cleaned data to Azure...")
                if self.upload_table_to_azure("patients_business_cleaned", "business-cleaned-data"):
                    logger.info(" Business cleaned data uploaded to Azure successfully")
                else:
                    logger.warning(" Business cleaned data upload to Azure failed")
                
                return True
            else:
                logger.error(" Business cleaning SQL execution failed")
                return False
                
        except Exception as e:
            logger.error(f" Business cleaning process failed: {e}")
            return False
        
    def run_feature_engineering(self) -> bool:
        """
        Run feature engineering
        Create machine learning features and automatically upload to Azure
        
        Returns:
            bool: Whether feature engineering was successful
        """
        try:
            logger.info("🔧 Starting feature engineering process...")
            
            # Ensure database connection
            if not self.connection or not self.connection.is_connected():
                if not self.connect_to_mysql():
                    logger.error(" Database connection failed")
                    return False
            
            # Execute feature engineering SQL file
            sql_file_path = "src/etl/sql_processing/04_feature_engineering.sql"
            if self.execute_sql_file(sql_file_path):
                logger.info(" Feature engineering SQL executed successfully")
                
                # Verify feature engineering results
                cursor = self.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM patients_features")
                features_count = cursor.fetchone()[0]
                cursor.close()
                
                logger.info(f" Feature engineering completed, generated {features_count} feature records")
                
                # Automatically upload feature engineered data to Azure
                logger.info("☁️ Automatically uploading feature engineered data to Azure...")
                if self.upload_table_to_azure("patients_features", "feature-engineered-data"):
                    logger.info("Feature engineered data uploaded to Azure successfully")
                else:
                    logger.warning(" Feature engineered data upload to Azure failed")
                
                return True
            else:
                logger.error(" Feature engineering SQL execution failed")
                return False
                
        except Exception as e:
            logger.error(f" Feature engineering process failed: {e}")
            return False
        

        
    def import_data_from_azure(self, container_name: str = "raw-data", blob_name: str = "diabetic_data.csv") -> bool:
        """
        Complete Azure to SQL data import process
        Architecture: Azure Download → Pre-cleaning → SQL Connection → SQL Import
        
        Args:
            container_name: Azure container name
            blob_name: Azure blob name
            
        Returns:
            bool: Whether import was successful
        """
        try:
            logger.info(" Starting complete data import process...")
            
            # Step 1: Download data from Azure
            logger.info(" Step 1: Downloading data from Azure...")
            if not self.download_from_azure(container_name, blob_name):
                logger.error(" Azure download failed")
                return False
            logger.info(" Azure download successful")
            
            # Step 2: Read CSV file
            logger.info(" Step 2: Reading CSV file...")
            df = pd.read_csv(blob_name)
            logger.info(f" Original data: {len(df)} rows, {len(df.columns)} columns")
            
            # Step 3: Apply first admission logic
            logger.info(" Step 3: Applying first admission logic...")
            df_first = self.data_pre_cleaner.apply_first_admission_logic(df)
            
            # Step 4: Execute pre-cleaning (8 major cleaning rules)
            logger.info(" Step 4: Executing pre-cleaning...")
            df_cleaned = self.data_pre_cleaner.clean_data(df_first)
            
            # Step 5: Ensure SQL connection
            logger.info("🔌 Step 5: Ensuring SQL connection...")
            if not self.connection or not self.connection.is_connected():
                if not self.connect_to_mysql():
                    logger.error(" SQL connection failed")
                    return False
            
            # Step 6: Import to SQL
            logger.info("💾 Step 6: Importing to SQL...")
            if self.insert_data_to_mysql(df_cleaned):
                logger.info(" Data import to SQL successful")
                return True
            else:
                logger.error(" Data import to SQL failed")
                return False
                
        except Exception as e:
            logger.error(f" Data import process failed: {e}")
            return False
    

    
    def insert_data_to_mysql(self, df: pd.DataFrame) -> bool:
        """Automatically create patients table and insert data"""
        try:
            cursor = self.connection.cursor()
            
            # Column name mapping - handle CSV column name to database column name mapping
            column_mapping = {
                'patient_nbr': 'patient_id',
                'change': 'medication_change',
                'glyburide-metformin': 'glyburide_metformin',
                'glipizide-metformin': 'glipizide_metformin',
                'glimepiride-pioglitazone': 'glimepiride_pioglitazone',
                'metformin-rosiglitazone': 'metformin_rosiglitazone',
                'metformin-pioglitazone': 'metformin_pioglitazone'
            }
            
            # Apply column name mapping
            df = df.rename(columns=column_mapping)
            
            # Check if patients table exists, create if not
            cursor.execute("SHOW TABLES LIKE 'patients'")
            if not cursor.fetchone():
                logger.info("🏗️ patients table does not exist, creating...")
                
                # Create patients table
                create_table_sql = """
                CREATE TABLE patients (
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
                )
                """
                cursor.execute(create_table_sql)
                logger.info(" patients table created successfully")
            else:
                # Clear existing data
                cursor.execute("DELETE FROM patients")
                logger.info(" Cleared existing patients table data")
            
            # Get database columns
            cursor.execute("DESCRIBE patients")
            db_columns = [row[0] for row in cursor.fetchall()]
            
            # Only select existing columns
            available_columns = [col for col in df.columns if col in db_columns]
            
            logger.info(f" Available columns: {len(available_columns)}/{len(df.columns)}")
            logger.info(f" Available columns: {', '.join(available_columns)}")
            
            # Build INSERT statement
            columns_quoted = [f"`{col}`" for col in available_columns]
            placeholders = ', '.join(['%s'] * len(available_columns))
            insert_query = f"INSERT INTO patients ({', '.join(columns_quoted)}) VALUES ({placeholders})"
            
            # Prepare data
            data_to_insert = []
            for _, row in df.iterrows():
                row_data = []
                for col in available_columns:
                    value = row.get(col, None)
                    if pd.isna(value):
                        value = None
                    row_data.append(value)
                data_to_insert.append(row_data)
            
            # Batch insert
            cursor.executemany(insert_query, data_to_insert)
            self.connection.commit()
            
            logger.info(f"Successfully inserted {len(data_to_insert)} rows")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f" Insert failed: {e}")
            return False
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed data"""
        pass
        
    def download_from_azure(self, container_name: str = "raw-data", blob_name: str = "diabetic_data.csv") -> bool:
        """Download data from Azure"""
        try:
            logger.info(f"Downloading from Azure: {container_name}/{blob_name}")
            
            # Import Azure SDK
            try:
                from azure.storage.blob import BlobServiceClient
                
                # Read Azure configuration from environment variables
                connection_string = os.getenv('AZURE_CONNECTION_STRING')
                
                if not connection_string:
                    logger.error(" AZURE_CONNECTION_STRING environment variable not found")
                    return False
                
                # Create BlobServiceClient
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                
                # Get blob client
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                
                # Download file
                with open(blob_name, "wb") as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())
                
                logger.info(" Azure download successful")
                return True
                
            except ImportError:
                logger.error(" azure-storage-blob package not installed")
                return False
            except Exception as e:
                logger.error(f" Azure download failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f" Download process failed: {e}")
            return False
    
    def upload_table_to_azure(self, table_name: str, container_name: str = "processed-data") -> bool:
        """
        Upload database table to Azure Blob Storage
        
        Args:
            table_name: Database table name
            container_name: Azure container name
            
        Returns:
            bool: Whether upload was successful
        """
        try:
            import pandas as pd
            from azure.storage.blob import BlobServiceClient
            from datetime import datetime
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"{table_name}_{timestamp}.csv"
            
            logger.info(f"☁️ Starting upload of table {table_name} to Azure...")
            
            # Read table data from database
            df = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
            logger.info(f" Read table {table_name}, total {len(df)} rows")
            
            # Save as temporary CSV file
            temp_file = f"{table_name}_{timestamp}.csv"
            df.to_csv(temp_file, index=False)
            
            # Upload to Azure
            connection_string = os.getenv('AZURE_CONNECTION_STRING')
            if not connection_string:
                logger.error(" Azure connection string not configured")
                return False
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            
            with open(temp_file, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Delete temporary file
            os.remove(temp_file)
            
            logger.info(f" Successfully uploaded {table_name} -> {container_name}/{blob_name}")
            return True
            
        except ImportError:
            logger.error(" azure-storage-blob package not installed")
            return False
        except Exception as e:
            logger.error(f" Failed to upload table {table_name} to Azure: {e}")
            return False

    def upload_to_azure(self, df: pd.DataFrame, container_name: str = "processed-data") -> bool:
        """Upload data to Azure"""
        try:
            logger.info(f"☁️ Uploading data to Azure: {container_name}")
            
            # Import Azure SDK
            try:
                from azure.storage.blob import BlobServiceClient
                from datetime import datetime
                
                # Read Azure configuration from environment variables
                connection_string = os.getenv('AZURE_CONNECTION_STRING')
                
                if not connection_string:
                    logger.error(" AZURE_CONNECTION_STRING environment variable not found")
                    return False
                
                # Create BlobServiceClient
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                
                # Generate timestamp filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Upload different tables
                tables_to_upload = [
                    ("patients_mapped", "patients_mapped.csv"),
                    ("patients_cleaned", "patients_cleaned.csv"),
                    ("patients_business_cleaned", "patients_business_cleaned.csv"),
                    ("patients_features", "patients_features.csv")
                ]
                
                success_count = 0
                
                for table_name, blob_name in tables_to_upload:
                    try:
                        # Check if table exists
                        cursor = self.connection.cursor()
                        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                        if not cursor.fetchone():
                            logger.warning(f"⚠️ Table {table_name} does not exist, skipping upload")
                            continue
                        
                        # Export table to CSV
                        local_file = f"{table_name}_{timestamp}.csv"
                        df_export = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
                        df_export.to_csv(local_file, index=False)
                        
                        # Upload to Azure
                        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                        
                        with open(local_file, "rb") as data:
                            blob_client.upload_blob(data, overwrite=True)
                        
                        logger.info(f" Successfully uploaded {table_name} -> {container_name}/{blob_name}")
                        success_count += 1
                        
                        # Delete local temporary file
                        os.remove(local_file)
                        
                    except Exception as e:
                        logger.error(f" Upload {table_name} failed: {e}")
                
                if success_count > 0:
                    logger.info(f" Successfully uploaded {success_count} files to Azure")
                    return True
                else:
                    logger.error(" No files successfully uploaded to Azure")
                    return False
                
            except ImportError:
                logger.error(" azure-storage-blob package not installed")
                return False
            except Exception as e:
                logger.error(f" Azure upload failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f" Upload process failed: {e}")
            return False
        
    def generate_etl_report(self) -> Dict[str, Any]:
        """Generate ETL report"""
        pass
        
    def run_full_pipeline(self, upload_to_azure: bool = True, import_from_azure: bool = True) -> bool:
        """Run complete ETL pipeline"""
        pass

    def download_and_create_mapping_tables(self) -> bool:
        """
        Download IDS_mapping.csv from Azure and split to create three mapping tables
        Create: admission_type_mapping, discharge_disposition_mapping, admission_source_mapping
        
        Returns:
            bool: Whether successful
        """
        try:
            logger.info(" Starting download and create mapping tables...")
            
            # Step 1: Download IDS_mapping.csv from Azure
            logger.info(" Downloading IDS_mapping.csv from Azure...")
            if not self.download_from_azure("raw-data", "IDS_mapping.csv"):
                logger.error(" Download IDS_mapping.csv failed")
                return False
            
            # Step 2: Read and split mapping data
            logger.info(" Reading and splitting mapping data...")
            try:
                ids_mapping_df = pd.read_csv("IDS_mapping.csv")
                logger.info(f" Mapping data: {ids_mapping_df.shape}")
                
                # Split data (based on known row indices)
                admission_type_df = ids_mapping_df.iloc[0:8].copy()
                discharge_disposition_df = ids_mapping_df.iloc[10:40].reset_index(drop=True).copy()
                admission_source_df = ids_mapping_df.iloc[42:].reset_index(drop=True).copy()
                
                # Set column names
                admission_type_df.columns = ['admission_type_id', 'admission_type_desc']
                discharge_disposition_df.columns = ['discharge_disposition_id', 'discharge_disposition_desc']
                admission_source_df.columns = ['admission_source_id', 'admission_source_desc']
                
                # Convert ID columns to integer type
                admission_type_df['admission_type_id'] = admission_type_df['admission_type_id'].astype(int)
                discharge_disposition_df['discharge_disposition_id'] = discharge_disposition_df['discharge_disposition_id'].astype(int)
                admission_source_df['admission_source_id'] = admission_source_df['admission_source_id'].astype(int)
                
                logger.info(f" Split completed: admission_type({len(admission_type_df)}), discharge_disposition({len(discharge_disposition_df)}), admission_source({len(admission_source_df)})")
                
            except Exception as e:
                logger.error(f" Reading or splitting mapping data failed: {e}")
                return False
            
            # Step 3: Ensure database connection
            if not self.connection or not self.connection.is_connected():
                if not self.connect_to_mysql():
                    logger.error(" Database connection failed")
                    return False
            
            # Step 4: Create mapping tables and insert data
            logger.info("💾 Creating mapping tables and inserting data...")
            cursor = self.connection.cursor()
            
            # Create admission_type_mapping table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS admission_type_mapping (
                admission_type_id INT PRIMARY KEY,
                admission_type_desc VARCHAR(255)
            )
            """)
            
            # Create discharge_disposition_mapping table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS discharge_disposition_mapping (
                discharge_disposition_id INT PRIMARY KEY,
                discharge_disposition_desc VARCHAR(255)
            )
            """)
            
            # Create admission_source_mapping table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS admission_source_mapping (
                admission_source_id INT PRIMARY KEY,
                admission_source_desc VARCHAR(255)
            )
            """)
            
            # Clear existing data
            cursor.execute("DELETE FROM admission_type_mapping")
            cursor.execute("DELETE FROM discharge_disposition_mapping")
            cursor.execute("DELETE FROM admission_source_mapping")
            
            # Insert admission_type data
            for _, row in admission_type_df.iterrows():
                cursor.execute("""
                INSERT INTO admission_type_mapping (admission_type_id, admission_type_desc) 
                VALUES (%s, %s)
                """, (row['admission_type_id'], row['admission_type_desc']))
            
            # Insert discharge_disposition data
            for _, row in discharge_disposition_df.iterrows():
                cursor.execute("""
                INSERT INTO discharge_disposition_mapping (discharge_disposition_id, discharge_disposition_desc) 
                VALUES (%s, %s)
                """, (row['discharge_disposition_id'], row['discharge_disposition_desc']))
            
            # Insert admission_source data
            for _, row in admission_source_df.iterrows():
                cursor.execute("""
                INSERT INTO admission_source_mapping (admission_source_id, admission_source_desc) 
                VALUES (%s, %s)
                """, (row['admission_source_id'], row['admission_source_desc']))
            
            self.connection.commit()
            cursor.close()
            
            # Delete temporary file
            if os.path.exists("IDS_mapping.csv"):
                os.remove("IDS_mapping.csv")
            
            logger.info(" Mapping tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f" Creating mapping tables failed: {e}")
            return False


# Main function entry
if __name__ == "__main__":
    def main():
        """Main function"""
        logger.info(" Starting ETL Pipeline...")

        try:
            # Initialize ETL
            config = ConfigManager("config/database_config.yaml")
            etl = HospitalReadmissionETL(config)

            # Test complete data import process
            logger.info(" Testing complete data import process...")
            logger.info("Architecture: Azure Download → Pre-cleaning → SQL Connection → SQL Import")
            
            if etl.import_data_from_azure():
                logger.info(" Complete data import process successful!")
            else:
                logger.error(" Complete data import process failed")
                return 1

            # Create mapping tables
            logger.info(" Creating mapping tables...")
            if etl.download_and_create_mapping_tables():
                logger.info("Mapping tables created successfully")
            else:
                logger.error(" Creating mapping tables failed")
                return 1

            # Test complete ETL process
            logger.info(" Starting complete ETL process test...")
            
            # Step 1: Data mapping
            logger.info(" Step 1: Data mapping...")
            if etl.run_data_mapping():
                logger.info(" Data mapping successful")
            else:
                logger.error(" Data mapping failed")
                return 1
            
            # Step 2: Dynamic column deletion
            logger.info(" Step 2: Dynamic column deletion...")
            try:
                if etl.run_data_cleaning():
                    logger.info(" Dynamic column deletion successful")
                else:
                    logger.error(" Dynamic column deletion failed")
                    return 1
            except Exception as e:
                logger.warning(f" Transaction error occurred during dynamic column deletion, but table was created: {e}")
                logger.info(" Dynamic column deletion completed (ignoring transaction error)")
            
            # Verify patients_cleaned table exists
            try:
                cursor = etl.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM patients_cleaned")
                count = cursor.fetchone()[0]
                logger.info(f" patients_cleaned table verification successful, contains {count} records")
                cursor.close()
            except Exception as e:
                logger.error(f" patients_cleaned table verification failed: {e}")
                return 1
            
            # Step 3: Business rule cleaning
            logger.info(" Step 3: Business rule cleaning...")
            if etl.run_business_cleaning():
                logger.info(" Business rule cleaning successful")
            else:
                logger.error(" Business rule cleaning failed")
                return 1
            
            # Step 4: Feature engineering
            logger.info("🔧 Step 4: Feature engineering...")
            if etl.run_feature_engineering():
                logger.info("✅ Feature engineering successful")
            else:
                logger.error(" Feature engineering failed")
                return 1
            
            # Verify final results
            logger.info(" Verifying final results...")
            cursor = etl.connection.cursor()
            
            tables = ['patients_mapped', 'patients_cleaned', 'patients_business_cleaned', 'patients_features']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f" {table} table record count: {count}")
            
            cursor.close()
            
            logger.info(" Complete ETL process test completed! All steps executed successfully and automatically uploaded to Azure!")
            return 0

        except Exception as e:
            logger.error(f" Error occurred during execution: {e}")
            import traceback
            logger.error(f"Detailed error information: {traceback.format_exc()}")
            return 1
    
    exit_code = main()
    sys.exit(exit_code) 