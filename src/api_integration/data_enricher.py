"""
Data Enricher for Hospital Readmission Project
Enriches cleaned data with external API information
"""

import pandas as pd
import mysql.connector
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .api_manager import APIManager, APIResponse
from .api_config import APIConfig

# Import utils modules
try:
    from utils.config import ConfigManager
    from data_ingestion.azure_uploader import AzureDataUploader
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.config import ConfigManager
    from src.data_ingestion.azure_uploader import AzureDataUploader

logger = logging.getLogger(__name__)

class DataEnricher:
    """Enriches hospital readmission data with external API information"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.api_manager = APIManager()
        self.azure_uploader = AzureDataUploader(self.config)
        self.mysql_config = self.config.get_mysql_config()
        self.connection = None
        self.cursor = None
    
    def connect_to_mysql(self) -> bool:
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.mysql_config.get('host', 'localhost'),
                port=self.mysql_config.get('port', 3306),
                database=self.mysql_config.get('database', 'hospital_readmission'),
                user=self.mysql_config.get('user', 'root'),
                password=self.mysql_config.get('password', 'hospital123'),
                charset=self.mysql_config.get('charset', 'utf8mb4')
            )
            self.cursor = self.connection.cursor()
            logger.info("✅ Connected to MySQL database for data enrichment")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MySQL: {e}")
            return False
    
    def disconnect_from_mysql(self):
        """Disconnect from MySQL database"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("🔌 Disconnected from MySQL database")
    
    def download_cleaned_data_from_azure(self) -> Optional[pd.DataFrame]:
        """Download cleaned data from Azure Data Lake"""
        try:
            logger.info("📥 Downloading cleaned data from Azure...")
            
            # Download patients_business_cleaned.csv from Azure
            blob_name = "patients_business_cleaned.csv"
            container_name = "processed-data"
            
            # Create temporary file path
            temp_file = f"/tmp/{blob_name}"
            
            # Download from Azure
            success = self.azure_uploader.download_blob_from_azure(
                container_name=container_name,
                blob_name=blob_name,
                local_file_path=temp_file
            )
            
            if not success:
                logger.error("❌ Failed to download cleaned data from Azure")
                return None
            
            # Read the CSV file
            df = pd.read_csv(temp_file)
            logger.info(f"✅ Downloaded {len(df)} records from Azure")
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error downloading cleaned data: {e}")
            return None
    
    def enrich_diagnosis_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich diagnosis data with ICD-9 API information"""
        logger.info("🔍 Enriching diagnosis data with ICD-9 API...")
        
        # Create new columns for enriched data
        df['diag_1_icd9_desc'] = None
        df['diag_2_icd9_desc'] = None
        df['diag_3_icd9_desc'] = None
        df['diag_1_icd9_category'] = None
        df['diag_2_icd9_category'] = None
        df['diag_3_icd9_category'] = None
        
        # Process unique diagnosis codes to avoid duplicate API calls
        unique_diagnoses = set()
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                unique_diagnoses.update(df[col].dropna().unique())
        
        # Cache for API responses
        diagnosis_cache = {}
        
        # Call ICD-9 API for each unique diagnosis
        for diagnosis in unique_diagnoses:
            if pd.isna(diagnosis) or diagnosis == '':
                continue
                
            response = self.api_manager.call_icd9_api(str(diagnosis))
            if response.success:
                diagnosis_cache[diagnosis] = response.data
            else:
                logger.warning(f"⚠️ Failed to get ICD-9 info for {diagnosis}: {response.error_message}")
        
        # Apply enriched data to dataframe
        for i, col in enumerate(['diag_1', 'diag_2', 'diag_3'], 1):
            if col in df.columns:
                for idx, diagnosis in df[col].items():
                    if diagnosis in diagnosis_cache:
                        api_data = diagnosis_cache[diagnosis]
                        # Handle ICD-9 API response format (list with display data)
                        if isinstance(api_data, list) and len(api_data) >= 4:
                            display_data = api_data[3]  # Display data is at index 3
                            if display_data and len(display_data) > 0:
                                for display_item in display_data:
                                    if len(display_item) >= 2:
                                        code_returned = display_item[0]
                                        name = display_item[1]
                                        # Check if this matches our diagnosis code
                                        if str(diagnosis) in code_returned:
                                            df.at[idx, f'diag_{i}_icd9_desc'] = name
                                            df.at[idx, f'diag_{i}_icd9_category'] = code_returned
                                            break
                        # Fallback to dict format if needed
                        elif isinstance(api_data, dict) and 'results' in api_data:
                            results = api_data['results']
                            if results and len(results) > 0:
                                result = results[0]
                                df.at[idx, f'diag_{i}_icd9_desc'] = result.get('name', '')
                                df.at[idx, f'diag_{i}_icd9_category'] = result.get('category', '')
        
        logger.info("✅ Diagnosis data enrichment completed")
        return df
    
    def enrich_medication_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich medication data with OpenFDA API information"""
        logger.info("💊 Enriching medication data with OpenFDA API...")
        
        # Identify medication columns
        medication_columns = [col for col in df.columns if any(med in col.lower() for med in 
                           ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'change'])]
        
        # Create new columns for enriched data
        for col in medication_columns:
            df[f'{col}_fda_info'] = None
            df[f'{col}_drug_class'] = None
            df[f'{col}_side_effects'] = None
        
        # Process unique medications
        unique_medications = set()
        for col in medication_columns:
            if col in df.columns:
                unique_medications.update(df[col].dropna().unique())
        
        # Cache for API responses
        medication_cache = {}
        
        # Call OpenFDA API for each unique medication
        for medication in unique_medications:
            if pd.isna(medication) or medication == '' or medication.lower() in ['no', 'unknown', 'none']:
                continue
                
            response = self.api_manager.call_openfda_api(str(medication))
            if response.success:
                medication_cache[medication] = response.data
            else:
                logger.warning(f"⚠️ Failed to get FDA info for {medication}: {response.error_message}")
        
        # Apply enriched data to dataframe
        for col in medication_columns:
            if col in df.columns:
                for idx, medication in df[col].items():
                    if medication in medication_cache:
                        api_data = medication_cache[medication]
                        if isinstance(api_data, dict) and 'results' in api_data:
                            results = api_data['results']
                            if results and len(results) > 0:
                                result = results[0]
                                df.at[idx, f'{col}_fda_info'] = result.get('openfda', {}).get('generic_name', [''])[0] if result.get('openfda', {}).get('generic_name') else ''
                                df.at[idx, f'{col}_drug_class'] = result.get('openfda', {}).get('pharm_class_cs', [''])[0] if result.get('openfda', {}).get('pharm_class_cs') else ''
                                df.at[idx, f'{col}_side_effects'] = result.get('warnings', [''])[0] if result.get('warnings') else ''
        
        logger.info("✅ Medication data enrichment completed")
        return df
    
    def enrich_patient_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich patient data with simulated FHIR API information"""
        logger.info("👤 Enriching patient data with FHIR API...")
        
        # Create new columns for enriched data
        df['fhir_patient_active'] = None
        df['fhir_patient_gender'] = None
        df['fhir_patient_birth_date'] = None
        df['fhir_patient_address_city'] = None
        df['fhir_patient_address_state'] = None
        df['fhir_patient_contact_phone'] = None
        
        # Process all patients for FHIR API
        sample_indices = df.index  # Process all records
        
        for idx in sample_indices:
            patient_id = df.at[idx, 'patient_id']
            response = self.api_manager.call_fhir_simulated_api(str(patient_id))
            
            if response.success:
                fhir_data = response.data
                df.at[idx, 'fhir_patient_active'] = fhir_data.get('active', False)
                df.at[idx, 'fhir_patient_gender'] = fhir_data.get('gender', 'unknown')
                df.at[idx, 'fhir_patient_birth_date'] = fhir_data.get('birthDate', '')
                
                # Extract address information
                if fhir_data.get('address') and len(fhir_data['address']) > 0:
                    address = fhir_data['address'][0]
                    df.at[idx, 'fhir_patient_address_city'] = address.get('city', '')
                    df.at[idx, 'fhir_patient_address_state'] = address.get('state', '')
                
                # Extract contact information
                if fhir_data.get('contact') and len(fhir_data['contact']) > 0:
                    contact = fhir_data['contact'][0]
                    if contact.get('telecom') and len(contact['telecom']) > 0:
                        telecom = contact['telecom'][0]
                        df.at[idx, 'fhir_patient_contact_phone'] = telecom.get('value', '')
        
        logger.info("✅ Patient data enrichment completed")
        return df
    
    def enrich_with_heterogeneous_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with heterogeneous sources (Lerner, Vendors, Research Teams)"""
        logger.info("🌐 Enriching data with heterogeneous sources...")
        
        # Create new columns for enriched data
        df['lerner_demographics'] = None
        df['vendor_insurance'] = None
        df['research_study_participant'] = None
        df['research_protocol'] = None
        
        # Process all records for heterogeneous APIs
        sample_indices = df.index  # Process all records
        
        for idx in sample_indices:
            patient_id = df.at[idx, 'patient_id']
            
            # Call Lerner Research Institute API
            lerner_response = self.api_manager.call_lerner_research_api(
                "/api/v1/patients/demographics",
                {"patient_id": patient_id}
            )
            if lerner_response.success:
                df.at[idx, 'lerner_demographics'] = str(lerner_response.data)
            
            # Call External Vendor API
            vendor_response = self.api_manager.call_external_vendor_api(
                "/api/v1/insurance",
                {"patient_id": patient_id}
            )
            if vendor_response.success:
                df.at[idx, 'vendor_insurance'] = str(vendor_response.data)
            
            # Call Research Teams API
            research_response = self.api_manager.call_research_teams_api(
                "/api/v1/studies/participants",
                {"patient_id": patient_id}
            )
            if research_response.success:
                df.at[idx, 'research_study_participant'] = str(research_response.data)
        
        logger.info("✅ Heterogeneous data enrichment completed")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality using data quality API"""
        logger.info("🔍 Validating data quality...")
        
        # Process all data for quality check
        sample_data = df.to_dict('records')
        
        quality_results = []
        for record in sample_data:
            response = self.api_manager.call_data_quality_api(record)
            if response.success:
                quality_results.append(response.data)
        
        # Aggregate quality results
        if quality_results:
            avg_quality_score = sum(result.get('quality_score', 0) for result in quality_results) / len(quality_results)
            all_issues = []
            for result in quality_results:
                all_issues.extend(result.get('issues', []))
            
            quality_summary = {
                'average_quality_score': avg_quality_score,
                'total_issues_found': len(all_issues),
                'common_issues': list(set(all_issues)),
                'recommendations': quality_results[0].get('recommendations', []) if quality_results else []
            }
        else:
            quality_summary = {
                'average_quality_score': 0.0,
                'total_issues_found': 0,
                'common_issues': [],
                'recommendations': []
            }
        
        logger.info(f"✅ Data quality validation completed. Score: {quality_summary['average_quality_score']:.2f}")
        return quality_summary
    
    def save_enriched_data_to_mysql(self, df: pd.DataFrame, table_name: str = "patients_api_enriched") -> bool:
        """Save enriched data to MySQL database"""
        try:
            logger.info(f"💾 Saving enriched data to MySQL table: {table_name}")
            
            # Create table if not exists
            # Handle column names with special characters by quoting them
            column_definitions = []
            for col in df.columns:
                if col != 'id':
                    # Quote column names that contain special characters
                    if any(char in col for char in ['-', ' ', '.', '/']):
                        column_definitions.append(f'`{col}` TEXT')
                    else:
                        column_definitions.append(f'{col} TEXT')
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                {', '.join(column_definitions)}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            self.cursor.execute(create_table_sql)
            
            # Clear existing data
            self.cursor.execute(f"TRUNCATE TABLE {table_name}")
            
            # Insert enriched data
            for _, row in df.iterrows():
                placeholders = ', '.join(['%s'] * len(row))
                # Quote column names that contain special characters
                quoted_columns = []
                for col in row.index:
                    if any(char in col for char in ['-', ' ', '.', '/']):
                        quoted_columns.append(f'`{col}`')
                    else:
                        quoted_columns.append(col)
                columns = ', '.join(quoted_columns)
                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                self.cursor.execute(insert_sql, tuple(row.values))
            
            self.connection.commit()
            logger.info(f"✅ Successfully saved {len(df)} enriched records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving enriched data to MySQL: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def upload_enriched_data_to_azure(self, df: pd.DataFrame, filename: str = "patients_api_enriched.csv") -> bool:
        """Upload enriched data to Azure Data Lake"""
        try:
            logger.info(f"☁️ Uploading enriched data to Azure: {filename}")
            
            # Create temporary file
            temp_file = f"/tmp/{filename}"
            df.to_csv(temp_file, index=False)
            
            # Upload to Azure
            success = self.azure_uploader.upload_csv_to_azure(
                local_file_path=temp_file,
                container_name="processed-data",
                blob_name=filename
            )
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if success:
                logger.info(f"✅ Successfully uploaded enriched data to Azure: {filename}")
                return True
            else:
                logger.error(f"❌ Failed to upload enriched data to Azure: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error uploading enriched data to Azure: {e}")
            return False
    
    def run_full_enrichment(self) -> bool:
        """Run complete data enrichment process"""
        try:
            logger.info("🚀 Starting full data enrichment process...")
            
            # Connect to MySQL
            if not self.connect_to_mysql():
                return False
            
            # Download cleaned data from Azure
            df = self.download_cleaned_data_from_azure()
            if df is None:
                logger.error("❌ Failed to download cleaned data")
                return False
            
            logger.info(f"📊 Processing {len(df)} records for enrichment")
            
            # Enrich data with various APIs
            df = self.enrich_diagnosis_data(df)
            df = self.enrich_medication_data(df)
            df = self.enrich_patient_data(df)
            df = self.enrich_with_heterogeneous_sources(df)
            
            # Validate data quality
            quality_summary = self.validate_data_quality(df)
            logger.info(f"📈 Data quality score: {quality_summary['average_quality_score']:.2f}")
            
            # Save enriched data to MySQL
            if not self.save_enriched_data_to_mysql(df):
                logger.error("❌ Failed to save enriched data to MySQL")
                return False
            
            # Upload enriched data to Azure
            if not self.upload_enriched_data_to_azure(df):
                logger.error("❌ Failed to upload enriched data to Azure")
                return False
            
            logger.info("🎉 Full data enrichment process completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in full enrichment process: {e}")
            return False
        finally:
            self.disconnect_from_mysql() 