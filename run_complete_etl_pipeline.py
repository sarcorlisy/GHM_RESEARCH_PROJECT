#!/usr/bin/env python3
"""
Complete ETL Data Cleaning Pipeline
From Azure download data → Pre-processing import SQL → Mapping → Cleaning → Business cleaning → Upload back to Azure
One file to complete all operations
"""

import sys
import os
import logging
import yaml
import pandas as pd
from datetime import datetime
import mysql.connector
from typing import Dict, List, Optional
import numpy as np # Added for np.nan

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteETLPipeline:
    def __init__(self, config_path: str = "config/database_config.yaml"):
        """Initialize complete ETL process"""
        self.config = self._load_config(config_path)
        self.connection = None
        # Generate timestamp, format: 20250802_01 (first run of today)
        today = datetime.now().strftime("%Y%m%d")
        
        # Check how many times it has been run today
        run_count = 1
        for filename in os.listdir('.'):
            if filename.startswith(f"diabetic_data_{today}_"):
                try:
                    count = int(filename.split('_')[-1].split('.')[0])
                    run_count = max(run_count, count + 1)
                except:
                    pass
        
        self.timestamp = f"{today}_{run_count:02d}"
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            raise
    
    def connect_database(self):
        """Connect to database"""
        try:
            mysql_config = self.config['database']['mysql']
            self.connection = mysql.connector.connect(
                host=mysql_config['host'],
                user=mysql_config['user'],
                password=mysql_config['password'],
                database=mysql_config['database'],
                port=mysql_config.get('port', 3306)
            )
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def download_from_azure(self, container_name: str, blob_name: str, local_file: str) -> bool:
        """Download file from Azure"""
        try:
            logger.info(f"Downloading from Azure: {container_name}/{blob_name} -> {local_file}")
            
            # Try to import Azure SDK
            try:
                from azure.storage.blob import BlobServiceClient
                import os
                
                # Read Azure configuration from config file
                azure_config_path = "config/azure_config.yaml"
                if os.path.exists(azure_config_path):
                    with open(azure_config_path, 'r') as f:
                        azure_config = yaml.safe_load(f)
                        connection_string = azure_config.get('azure', {}).get('storage_account', {}).get('connection_string')
                else:
                    logger.error("Azure configuration file does not exist: config/azure_config.yaml")
                    return False
                
                if not connection_string:
                    logger.error("Azure storage connection string not found")
                    return False
                
                # Create BlobServiceClient
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                
                # Get container client
                container_client = blob_service_client.get_container_client(container_name)
                
                # Get blob client
                blob_client = container_client.get_blob_client(blob_name)
                
                # Download file
                with open(local_file, "wb") as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())
                
                logger.info(f"File download successful: {local_file}")
                return True
                
            except ImportError:
                logger.error("Azure SDK not installed, please run: pip install azure-storage-blob")
                return False
            except Exception as e:
                logger.error(f"Azure download failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Azure download failed: {e}")
            return False
    
    def upload_to_azure(self, local_file: str, container_name: str, blob_name: str) -> bool:
        """Upload file to Azure"""
        try:
            logger.info(f"Uploading to Azure: {local_file} -> {container_name}/{blob_name}")
            
            # Try to import Azure SDK
            try:
                from azure.storage.blob import BlobServiceClient
                import os
                
                # Read Azure configuration from config file
                azure_config_path = "config/azure_config.yaml"
                if os.path.exists(azure_config_path):
                    with open(azure_config_path, 'r') as f:
                        azure_config = yaml.safe_load(f)
                        connection_string = azure_config.get('azure', {}).get('storage_account', {}).get('connection_string')
                else:
                    logger.error("Azure configuration file does not exist: config/azure_config.yaml")
                    return False
                
                if not connection_string:
                    logger.error("Azure storage connection string not found")
                    return False
                
                # Create BlobServiceClient
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                
                # Get container client
                container_client = blob_service_client.get_container_client(container_name)
                
                # Get blob client
                blob_client = container_client.get_blob_client(blob_name)
                
                # Upload file
                with open(local_file, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                logger.info(f"File upload successful: {blob_name}")
                return True
                
            except ImportError:
                logger.error("Azure SDK not installed, please run: pip install azure-storage-blob")
                return False
            except Exception as e:
                logger.error(f"Azure upload failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Azure upload failed: {e}")
            return False
    
    def import_data_to_mysql(self, csv_file: str, table_name: str) -> bool:
        """Import CSV data to MySQL, apply eight major data cleaning principles"""
        try:
            logger.info(f"Importing data to MySQL: {csv_file} -> {table_name}")
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            logger.info(f"Read data: {len(df)} rows, {len(df.columns)} columns")
            
            # Apply eight major data cleaning principles
            logger.info("Applying eight major data cleaning principles...")
            
            # 1. Keep first admission record for each patient
            logger.info(f"Original record count: {len(df)}")
            df_sorted = df.sort_values(by='encounter_id')
            df = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')
            logger.info(f"After first admission filtering: {len(df)} rows")
            
            # 2. Delete records of patients who died or received hospice care
            hospice_or_death_ids = [11, 13, 14, 19, 20, 21]
            records_before_filter = len(df)
            df = df[~df['discharge_disposition_id'].isin(hospice_or_death_ids)]
            logger.info(f"Deleted death/hospice patients: {records_before_filter - len(df)} records")
            logger.info(f"Filtered record count: {len(df)}")
            
            # 3. Handle missing values
            df = df.fillna('Unknown')
            logger.info(" Missing value processing completed")
            
            # 4. Handle special characters
            df = df.replace('?', 'Unknown')
            logger.info(" Special character processing completed")
            
            # 5. Ensure patient_id is string
            if 'patient_id' in df.columns:
                df['patient_id'] = df['patient_id'].astype(str)
            
            # 6. Process age field - extract midpoint of age range
            if 'age' in df.columns:
                def extract_age_midpoint(age_str):
                    if pd.isna(age_str) or age_str == 'Unknown':
                        return None
                    try:
                        # Handle formats like '[70-80)' or '[0-10)' etc.
                        if isinstance(age_str, str) and '[' in age_str and ')' in age_str:
                            # Extract number range
                            age_range = age_str.replace('[', '').replace(')', '')
                            if '-' in age_range:
                                start, end = age_range.split('-')
                                return int((int(start) + int(end)) / 2)
                            else:
                                return int(age_range)
                        else:
                            return int(age_str)
                    except:
                        return None
                
                df['age'] = df['age'].apply(extract_age_midpoint)
                logger.info(" Age field processing completed")
            
            # 7. Process numeric fields - ensure correct types
            numeric_columns = [
                'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0).astype(int)
            
            logger.info(" Numeric field processing completed")
            
            # 8. Process medication fields - standardize
            medication_columns = [
                'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
            ]
            
            for col in medication_columns:
                if col in df.columns:
                    df[col] = df[col].str.upper()
                    df[col] = df[col].replace(['UNKNOWN', 'NONE'], 'No')
            
            logger.info(" Medication field standardization completed")
            
            # 9. Process diagnosis fields - clean format
            diagnosis_columns = ['diag_1', 'diag_2', 'diag_3']
            for col in diagnosis_columns:
                if col in df.columns:
                    df[col] = df[col].str.strip()
                    df[col] = df[col].replace('', 'Unknown')
            
            logger.info(" Diagnosis field processing completed")
            
            # 10. Data quality check
            total_rows = len(df)
            null_counts = df.isnull().sum()
            logger.info(f" Data cleaning completed - Total rows: {total_rows}")
            logger.info(f" Null value statistics by column: {null_counts.sum()} null values")
            
            # 11. Add created_at timestamp
            df['created_at'] = pd.Timestamp.now()
            
            logger.info(f"Final cleaned data: {len(df)} rows, {len(df.columns)} columns")
            
            # Create table
            cursor = self.connection.cursor()
            
            # Delete existing table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table structure
            columns = []
            for col in df.columns:
                col_type = "TEXT"  # Default to TEXT type
                if df[col].dtype == 'int64':
                    col_type = "INT"
                elif df[col].dtype == 'float64':
                    col_type = "FLOAT"
                elif df[col].dtype == 'datetime64[ns]':
                    col_type = "DATETIME"
                columns.append(f"`{col}` {col_type}")
            
            create_table_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(columns)}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cursor.execute(create_table_sql)
            
            # Insert data
            for _, row in df.iterrows():
                placeholders = ', '.join(['%s'] * len(row))
                insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                cursor.execute(insert_sql, tuple(row.values))
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Data import successful: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Data import failed: {e}")
            return False
    
    def create_mapping_tables(self):
        """Create mapping tables"""
        try:
            logger.info("Creating mapping tables...")
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
            
            # Insert mapping data (using sample data here)
            # In actual projects, should read from IDS_mapping.csv
            cursor.execute("""
            INSERT IGNORE INTO admission_type_mapping VALUES 
            (1, 'Emergency'), (2, 'Urgent'), (3, 'Elective'), (4, 'Newborn'), (5, 'Not Available'),
            (6, 'NULL'), (7, 'Trauma Center'), (8, 'Not Mapped')
            """)
            
            cursor.execute("""
            INSERT IGNORE INTO discharge_disposition_mapping VALUES 
            (1, 'Discharged to home'), (2, 'Discharged/transferred to another short term hospital'),
            (3, 'Discharged/transferred to SNF'), (6, 'Discharged/transferred to home with home health service'),
            (8, 'Discharged/transferred to home under care of Home IV provider'), (9, 'Admitted as an inpatient to this hospital'),
            (13, 'Hospice / home'), (14, 'Hospice / medical facility'), (19, 'Expired at home. Medicaid only, hospice.'),
            (20, 'Expired in a medical facility. Medicaid only, hospice.'), (21, 'Expired, place unknown. Medicaid only, hospice.'),
            (11, 'Expired'), (25, 'Not Mapped'), (26, 'Unknown/Invalid')
            """)
            
            cursor.execute("""
            INSERT IGNORE INTO admission_source_mapping VALUES 
            (1, 'Physician Referral'), (2, 'Clinic Referral'), (3, 'HMO Referral'), (4, 'Transfer from a hospital'),
            (5, 'Transfer from a Skilled Nursing Facility (SNF)'), (6, 'Transfer from another health care facility'),
            (7, 'Emergency Room'), (8, 'Court/Law Enforcement'), (9, 'Not Available'), (17, 'Transfer from clinic'),
            (20, 'Transfer from hospital inpt/same fac reslt in a sep claim'), (21, 'Transfer from hospital inpt/same fac reslt in a sep claim'),
            (22, 'Transfer from hospital inpt/same fac reslt in a sep claim'), (25, 'Not Mapped'), (26, 'Unknown/Invalid')
            """)
            
            self.connection.commit()
            cursor.close()
            logger.info("Mapping tables created successfully")
            
        except Exception as e:
            logger.error(f"Creating mapping tables failed: {e}")
            raise
    
    def create_mapped_data(self):
        """Create mapped data table"""
        try:
            logger.info("Creating mapped data table...")
            
            # Use timestamp to create table name
            mapped_table_name = f"patients_mapped_{self.timestamp}"
            
            cursor = self.connection.cursor()
            
            # Create mapped data table
            sql = f"""
            CREATE TABLE IF NOT EXISTS {mapped_table_name} AS
            SELECT 
                p.*,
                atm.admission_type_desc,
                ddm.discharge_disposition_desc,
                asm.admission_source_desc
            FROM patients p
            LEFT JOIN admission_type_mapping atm ON p.admission_type_id = atm.admission_type_id
            LEFT JOIN discharge_disposition_mapping ddm ON p.discharge_disposition_id = ddm.discharge_disposition_id
            LEFT JOIN admission_source_mapping asm ON p.admission_source_id = asm.admission_source_id
            """
            
            cursor.execute(sql)
            self.connection.commit()
            
            # Rename to standard table name
            cursor.execute(f"DROP TABLE IF EXISTS patients_mapped")
            cursor.execute(f"RENAME TABLE {mapped_table_name} TO patients_mapped")
            self.connection.commit()
            
            cursor.close()
            logger.info("Mapped data table created successfully")
            
        except Exception as e:
            logger.error(f"Creating mapped data table failed: {e}")
            raise
    
    def create_cleaned_data(self):
        """Create cleaned data table"""
        try:
            logger.info("Creating cleaned data table...")
            
            # Use timestamp to create table name
            cleaned_table_name = f"patients_cleaned_{self.timestamp}"
            
            cursor = self.connection.cursor()
            
            # Create cleaned data table (remove mapped ID columns)
            sql = f"""
            CREATE TABLE IF NOT EXISTS {cleaned_table_name} AS
            SELECT 
                encounter_id, patient_nbr, race, gender, age, weight,
                time_in_hospital, payer_code, medical_specialty,
                num_lab_procedures, num_procedures, num_medications,
                number_outpatient, number_emergency, number_inpatient,
                diag_1, diag_2, diag_3, number_diagnoses,
                max_glu_serum, A1Cresult,
                metformin, repaglinide, nateglinide, chlorpropamide,
                glimepiride, acetohexamide, glipizide, glyburide,
                tolbutamide, pioglitazone, rosiglitazone, acarbose,
                miglitol, troglitazone, tolazamide, examide,
                citoglipton, insulin, `glyburide-metformin`,
                `glipizide-metformin`, `glimepiride-pioglitazone`,
                `metformin-rosiglitazone`, `metformin-pioglitazone`,
                `change`, diabetesMed, readmitted, created_at,
                admission_type_desc, discharge_disposition_desc, admission_source_desc
            FROM patients_mapped
            """
            
            cursor.execute(sql)
            self.connection.commit()
            
            # Rename to standard table name
            cursor.execute(f"DROP TABLE IF EXISTS patients_cleaned")
            cursor.execute(f"RENAME TABLE {cleaned_table_name} TO patients_cleaned")
            self.connection.commit()
            
            cursor.close()
            logger.info("Cleaned data table created successfully")
            
        except Exception as e:
            logger.error(f"Creating cleaned data table failed: {e}")
            raise

    def run_dynamic_column_cleaning(self):
        """Run dynamic column cleaning"""
        try:
            logger.info("Starting dynamic column cleaning...")
            
            # Create invalid value analysis table
            cursor = self.connection.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS invalid_value_analysis AS
            SELECT 
                'weight' as column_name,
                COUNT(CASE WHEN weight IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
                COUNT(*) as total_count,
                ROUND(COUNT(CASE WHEN weight IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
            FROM patients_cleaned
            UNION ALL
            SELECT 
                'max_glu_serum' as column_name,
                COUNT(CASE WHEN max_glu_serum IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
                COUNT(*) as total_count,
                ROUND(COUNT(CASE WHEN max_glu_serum IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
            FROM patients_cleaned
            UNION ALL
            SELECT 
                'A1Cresult' as column_name,
                COUNT(CASE WHEN A1Cresult IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) as invalid_count,
                COUNT(*) as total_count,
                ROUND(COUNT(CASE WHEN A1Cresult IN ('Unknown', 'Not Available', 'NULL', '') THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_percentage
            FROM patients_cleaned
            """)
            
            # Get columns to delete
            cursor.execute("""
            SELECT column_name FROM invalid_value_analysis 
            WHERE invalid_percentage > 50
            """)
            columns_to_remove = [row[0] for row in cursor.fetchall()]
            
            # Get all columns
            cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'patients_cleaned' AND TABLE_SCHEMA = DATABASE()
            ORDER BY ORDINAL_POSITION
            """)
            all_columns = [row[0] for row in cursor.fetchall()]
            
            # Filter columns to keep
            columns_to_keep = [col for col in all_columns if col not in columns_to_remove]
            
            # Create new cleaned table
            columns_str = ', '.join([f"`{col}`" for col in columns_to_keep])
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS patients_cleaned_updated AS
            SELECT {columns_str}
            FROM patients_cleaned
            """)
            
            # Delete original table and rename
            cursor.execute("DROP TABLE IF EXISTS patients_cleaned")
            cursor.execute("RENAME TABLE patients_cleaned_updated TO patients_cleaned")
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Dynamic column cleaning completed! Deleted {len(columns_to_remove)} columns: {columns_to_remove}")
            
        except Exception as e:
            logger.error(f"Dynamic column cleaning failed: {e}")
            raise

    def create_business_cleaned_data(self):
        """Create business cleaned data table"""
        try:
            logger.info("Creating business cleaned data table...")
            
            # Use timestamp to create table name
            business_cleaned_table_name = f"patients_business_cleaned_{self.timestamp}"
            
            cursor = self.connection.cursor()
            
            # Create business cleaned data table (remove patients who cannot be readmitted)
            sql = f"""
            CREATE TABLE IF NOT EXISTS {business_cleaned_table_name} AS
            SELECT *
            FROM patients_cleaned
            WHERE discharge_disposition_desc NOT IN (
                'Expired', 'Hospice / home', 'Hospice / medical facility',
                'Expired at home. Medicaid only, hospice.',
                'Expired in a medical facility. Medicaid only, hospice.',
                'Expired, place unknown. Medicaid only, hospice.'
            )
            """
            
            cursor.execute(sql)
            self.connection.commit()
            
            # Create indexes to improve query performance
            try:
                cursor.execute(f"CREATE INDEX idx_patient_id_{self.timestamp} ON {business_cleaned_table_name} (patient_id)")
                cursor.execute(f"CREATE INDEX idx_encounter_id_{self.timestamp} ON {business_cleaned_table_name} (encounter_id)")
                cursor.execute(f"CREATE INDEX idx_readmitted_{self.timestamp} ON {business_cleaned_table_name} (readmitted)")
                self.connection.commit()
            except Exception as e:
                logger.warning(f"Warning when creating indexes (may already exist): {e}")
            
            # Rename to standard table name
            cursor.execute(f"DROP TABLE IF EXISTS patients_business_cleaned")
            cursor.execute(f"RENAME TABLE {business_cleaned_table_name} TO patients_business_cleaned")
            self.connection.commit()
            
            cursor.close()
            logger.info("Business cleaned data table created successfully")
            
        except Exception as e:
            logger.error(f"Creating business cleaned data table failed: {e}")
            raise
    
    def export_table_to_csv(self, table_name: str, csv_file: str) -> bool:
        """Export table to CSV file"""
        try:
            logger.info(f"Exporting table to CSV: {table_name} -> {csv_file}")
            
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Get column names
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            columns = [row[0] for row in cursor.fetchall()]
            
            # Create DataFrame and save
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(csv_file, index=False)
            
            cursor.close()
            logger.info(f"Table export successful: {csv_file} ({len(df)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Table export failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run complete ETL process"""
        try:
            logger.info("="*60)
            logger.info("Starting complete ETL data cleaning process")
            logger.info("="*60)
            
            # Step 1: Download data from Azure
            logger.info("\nStep 1: Download data from Azure")
            local_diabetic_data = f"diabetic_data_{self.timestamp}.csv"
            if not self.download_from_azure("raw-data", "diabetic_data.csv", local_diabetic_data):
                logger.warning("Azure download failed, using local file")
                local_diabetic_data = "data/diabetic_data.csv"
            
            # Step 2: Import data to MySQL (re-import every time)
            logger.info("\nStep 2: Import data to MySQL")
            # First delete old tables, recreate (note foreign key constraint order)
            cursor = self.connection.cursor()
            # Delete tables according to foreign key dependency: medications -> encounters -> patients
            cursor.execute("DROP TABLE IF EXISTS patients_business_cleaned")
            cursor.execute("DROP TABLE IF EXISTS patients_cleaned")
            cursor.execute("DROP TABLE IF EXISTS patients_mapped")
            cursor.execute("DROP TABLE IF EXISTS medications")  # Delete table that references encounters first
            cursor.execute("DROP TABLE IF EXISTS encounters")   # Delete table that references patients next
            cursor.execute("DROP TABLE IF EXISTS patients")     # Delete referenced table last
            self.connection.commit()
            cursor.close()
            
            if not self.import_data_to_mysql(local_diabetic_data, "patients"):
                raise Exception("Data import failed")
            
            # Step 3: Create mapping tables
            logger.info("\nStep 3: Create mapping tables")
            self.create_mapping_tables()
            
            # Step 4: Create mapped data
            logger.info("\nStep 4: Create mapped data")
            self.create_mapped_data()
            
            # Step 5: Upload mapped data to Azure
            logger.info("\nStep 5: Upload mapped data to Azure")
            mapped_data_csv = f"mapped_data_{self.timestamp}.csv"
            if self.export_table_to_csv("patients_mapped", mapped_data_csv):
                self.upload_to_azure(mapped_data_csv, "processed-data", "mapped_data.csv")
            
            # Step 6: Create cleaned data
            logger.info("\nStep 6: Create cleaned data")
            self.create_cleaned_data()
            
            # Step 7: Run dynamic column cleaning
            logger.info("\nStep 7: Run dynamic column cleaning")
            self.run_dynamic_column_cleaning()
            
            # Step 8: Upload cleaned data to Azure
            logger.info("\nStep 8: Upload cleaned data to Azure")
            cleaned_data_csv = f"patients_cleaned_{self.timestamp}.csv"
            if self.export_table_to_csv("patients_cleaned", cleaned_data_csv):
                self.upload_to_azure(cleaned_data_csv, "processed-data", "patients_cleaned.csv")
            
            # Step 9: Create business cleaned data
            logger.info("\nStep 9: Create business cleaned data")
            self.create_business_cleaned_data()
            
            # Step 10: Upload business cleaned data to Azure
            logger.info("\nStep 10: Upload business cleaned data to Azure")
            business_cleaned_csv = f"patients_business_cleaned_{self.timestamp}.csv"
            if self.export_table_to_csv("patients_business_cleaned", business_cleaned_csv):
                self.upload_to_azure(business_cleaned_csv, "processed-data", "patients_business_cleaned.csv")
            
            # Step 11: Generate report
            logger.info("\nStep 11: Generate processing report")
            self.generate_report()
            
            logger.info("\n" + "="*60)
            logger.info("Complete ETL data cleaning process executed successfully!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"ETL process execution failed: {e}")
            raise
    
    def generate_report(self):
        """Generate processing report"""
        try:
            cursor = self.connection.cursor()
            
            # Get record counts for each table
            tables = ['patients', 'patients_mapped', 'patients_cleaned', 'patients_business_cleaned']
            report_data = []
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                report_data.append((table, count))
            
            cursor.close()
            
            # Generate report
            report_file = f"etl_report_{self.timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write("ETL Data Cleaning Process Report\n")
                f.write("="*50 + "\n")
                f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Record counts by table:\n")
                for table, count in report_data:
                    f.write(f"  {table}: {count:,} rows\n")
                
                f.write(f"\nGenerated files:\n")
                f.write(f"  - mapped_data_{self.timestamp}.csv\n")
                f.write(f"  - patients_cleaned_{self.timestamp}.csv\n")
                f.write(f"  - patients_business_cleaned_{self.timestamp}.csv\n")
            
            logger.info(f"Processing report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Generating report failed: {e}")
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def main():
    """Main function"""
    try:
        # Create complete ETL process
        pipeline = CompleteETLPipeline()
        
        # Connect to database
        pipeline.connect_database()
        
        # Run complete process
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("✅ Complete ETL data cleaning process executed successfully!")
        print("="*60)
        print("Processed files:")
        print(f"  - mapped_data_{pipeline.timestamp}.csv")
        print(f"  - patients_cleaned_{pipeline.timestamp}.csv")
        print(f"  - patients_business_cleaned_{pipeline.timestamp}.csv")
        print(f"  - etl_report_{pipeline.timestamp}.txt")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.close_connection()

if __name__ == "__main__":
    main() 