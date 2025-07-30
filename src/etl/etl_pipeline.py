"""
ETL Pipeline for Hospital Readmission Data
Coordinates the entire data processing workflow
"""

import os
import pandas as pd
import mysql.connector
from typing import Dict, Any, Optional
import logging

from ..utils.config import ConfigManager
from ..utils.logging_config import get_logger, PipelineLogger
from ..data_ingestion.azure_uploader import AzureDataUploader
import sys
import os

# Add parent directory to path for import_data_to_mysql
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = get_logger(__name__)

class HospitalReadmissionETL:
    """Main ETL pipeline for hospital readmission data"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.mysql_config = self.config.get_mysql_config()
        self.azure_uploader = AzureDataUploader(self.config)
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
            logger.info("✅ Connected to MySQL database")
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
    
    def import_data_from_azure(self) -> bool:
        """
        从Azure导入原始数据到MySQL
        使用修改后的import_data_to_mysql.py逻辑
        """
        try:
            logger.info("☁️ 开始从Azure导入数据到MySQL...")
            
            # 导入DataImporter类
            from import_data_to_mysql import DataImporter
            
            # 创建DataImporter实例
            importer = DataImporter(
                host=self.mysql_config.get('host', 'localhost'),
                database=self.mysql_config.get('database', 'hospital_readmission'),
                user=self.mysql_config.get('user', 'root'),
                password=self.mysql_config.get('password', 'hospital123')
            )
            
            # 连接到数据库
            if not importer.connect():
                logger.error("❌ 无法连接到MySQL数据库")
                return False
            
            try:
                # 清空现有表
                logger.info("🗑️ 清空现有表...")
                if not importer.clear_tables():
                    logger.error("❌ 清空表失败")
                    return False
                
                # 重新创建表
                logger.info("🏗️ 重新创建表结构...")
                if not importer.recreate_tables():
                    logger.error("❌ 创建表失败")
                    return False
                
                # 从Azure导入数据（应用首次入院逻辑）
                logger.info("📥 从Azure导入数据...")
                if not importer.import_csv_data(use_azure=True):
                    logger.error("❌ 数据导入失败")
                    return False
                
                logger.info("✅ 从Azure导入数据成功")
                return True
                
            finally:
                importer.disconnect()
                
        except Exception as e:
            logger.error(f"❌ 从Azure导入数据失败: {e}")
            return False
    
    def execute_sql_file(self, sql_file_path: str) -> bool:
        """Execute SQL file from the sql_processing directory"""
        try:
            # Read SQL file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    try:
                        self.cursor.execute(statement)
                        # Fetch results if it's a SELECT statement to avoid "Unread result found"
                        if statement.strip().upper().startswith('SELECT'):
                            self.cursor.fetchall()
                        logger.info(f"✅ Executed SQL statement: {statement[:50]}...")
                    except Exception as e:
                        logger.warning(f"⚠️ Statement failed: {statement[:50]}... Error: {e}")
                        # Continue with next statement
            
            self.connection.commit()
            logger.info(f"✅ Successfully executed SQL file: {sql_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL file {sql_file_path}: {e}")
            return False
    
    def run_data_cleaning(self) -> bool:
        """Run data cleaning SQL script"""
        logger.info("🧹 Starting data cleaning process...")
        
        sql_file = "src/etl/sql_processing/01_data_cleaning.sql"
        if os.path.exists(sql_file):
            success = self.execute_sql_file(sql_file)
            if success:
                logger.info("✅ Data cleaning completed successfully")
                return True
            else:
                logger.error("❌ Data cleaning failed")
                return False
        else:
            logger.error(f"❌ SQL file not found: {sql_file}")
            return False
    
    def run_data_mapping(self) -> bool:
        """Run data mapping SQL script"""
        logger.info("🗺️ Starting data mapping process...")
        
        sql_file = "src/etl/sql_processing/03_data_mapping.sql"
        if os.path.exists(sql_file):
            success = self.execute_sql_file(sql_file)
            if success:
                logger.info("✅ Data mapping completed successfully")
                return True
            else:
                logger.error("❌ Data mapping failed")
                return False
        else:
            logger.error(f"❌ SQL file not found: {sql_file}")
            return False
    

    
    def run_feature_engineering(self) -> bool:
        """Run feature engineering SQL script"""
        logger.info("🔧 Starting feature engineering process...")
        
        sql_file = "src/etl/sql_processing/02_feature_engineering.sql"
        if os.path.exists(sql_file):
            success = self.execute_sql_file(sql_file)
            if success:
                logger.info("✅ Feature engineering completed successfully")
                return True
            else:
                logger.error("❌ Feature engineering failed")
                return False
        else:
            logger.error(f"❌ SQL file not found: {sql_file}")
            return False
    
    def get_processed_data(self, data_type: str = "mapped") -> Optional[pd.DataFrame]:
        """Get processed data from MySQL for Python ML pipeline"""
        try:
            if data_type == "mapped":
                # Get mapped data with descriptions
                query = """
                SELECT 
                    patient_id,
                    encounter_id,
                    race,
                    gender,
                    age,
                    weight,
                    admission_type_id,
                    admission_type_desc,
                    discharge_disposition_id,
                    discharge_disposition_desc,
                    admission_source_id,
                    admission_source_desc,
                    time_in_hospital,
                    payer_code,
                    medical_specialty,
                    num_lab_procedures,
                    num_procedures,
                    num_medications,
                    number_outpatient,
                    number_emergency,
                    number_inpatient,
                    diag_1,
                    diag_2,
                    diag_3,
                    number_diagnoses,
                    max_glu_serum,
                    A1Cresult,
                    metformin,
                    repaglinide,
                    nateglinide,
                    chlorpropamide,
                    glimepiride,
                    acetohexamide,
                    glipizide,
                    glyburide,
                    tolbutamide,
                    pioglitazone,
                    rosiglitazone,
                    acarbose,
                    miglitol,
                    troglitazone,
                    tolazamide,
                    examide,
                    citoglipton,
                    insulin,
                    `glyburide-metformin`,
                    `glipizide-metformin`,
                    `glimepiride-pioglitazone`,
                    `metformin-rosiglitazone`,
                    `metformin-pioglitazone`,
                    medication_change,
                    diabetesMed,
                    readmitted
                FROM patients_mapped
                WHERE patient_id IS NOT NULL
                """

            else:
                # Get feature engineered data (original logic)
                query = """
                SELECT 
                    patient_id,
                    age_cleaned,
                    gender_cleaned,
                    race_cleaned,
                    age_group,
                    time_in_hospital_cleaned,
                    stay_duration_category,
                    num_lab_procedures_cleaned,
                    num_procedures_cleaned,
                    num_medications_cleaned,
                    number_outpatient_cleaned,
                    number_emergency_cleaned,
                    number_inpatient_cleaned,
                    total_previous_visits,
                    visit_frequency_category,
                    number_diagnoses_cleaned,
                    diagnosis_complexity,
                    admission_type_id_cleaned,
                    discharge_disposition_id_cleaned,
                    admission_source_id_cleaned,
                    readmission_30_days,
                    readmission_any,
                    risk_score,
                    risk_category
                FROM patients_features
                WHERE patient_id IS NOT NULL
                """
            
            df = pd.read_sql(query, self.connection)
            logger.info(f"✅ Retrieved {len(df)} processed records from database")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve processed data: {e}")
            return None
    
    def upload_to_azure(self, df: pd.DataFrame, container_name: str = "processed-data") -> bool:
        """Upload processed data to Azure Data Lake"""
        logger.info("☁️ Uploading processed data to Azure...")
        
        # Upload as CSV
        csv_success = self.azure_uploader.upload_dataframe_to_azure(
            df=df,
            container_name=container_name,
            blob_name="hospital_readmission_processed.csv",
            file_format="csv"
        )
        
        # Upload as Parquet (more efficient for large datasets)
        parquet_success = self.azure_uploader.upload_dataframe_to_azure(
            df=df,
            container_name=container_name,
            blob_name="hospital_readmission_processed.parquet",
            file_format="parquet"
        )
        
        if csv_success and parquet_success:
            logger.info("✅ Successfully uploaded processed data to Azure")
            return True
        else:
            logger.warning("⚠️ Partial upload to Azure completed")
            return False
    
    def generate_etl_report(self) -> Dict[str, Any]:
        """Generate ETL processing report"""
        try:
            # Get data quality summary
            quality_query = "SELECT metric, value FROM data_quality_summary"
            quality_df = pd.read_sql(quality_query, self.connection)
            
            # Get feature summary
            feature_query = "SELECT feature_group, metric, value FROM feature_summary_stats"
            feature_df = pd.read_sql(feature_query, self.connection)
            
            # Get table row counts
            tables = ['patients', 'patients_cleaned', 'patients_features']
            table_counts = {}
            
            for table in tables:
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {table}"
                    count_df = pd.read_sql(count_query, self.connection)
                    table_counts[table] = count_df['count'].iloc[0]
                except:
                    table_counts[table] = 0
            
            report = {
                'data_quality': quality_df.to_dict('records'),
                'feature_summary': feature_df.to_dict('records'),
                'table_counts': table_counts,
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info("📊 ETL report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate ETL report: {e}")
            return {}
    
    def run_full_pipeline(self, upload_to_azure: bool = True, import_from_azure: bool = True) -> bool:
        """
        Run the complete ETL pipeline
        
        Args:
            upload_to_azure: Whether to upload results to Azure
            import_from_azure: Whether to import data from Azure (vs local file)
        """
        with PipelineLogger("hospital_readmission_etl") as pipeline_logger:
            try:
                # Step 1: Import data from Azure (if requested)
                if import_from_azure:
                    logger.info("☁️ Step 1: Importing data from Azure...")
                    if not self.import_data_from_azure():
                        logger.error("❌ Failed to import data from Azure")
                        return False
                    logger.info("✅ Data imported from Azure successfully")
                else:
                    logger.info("📁 Step 1: Skipping Azure import, using existing data")
                
                # Step 2: Connect to database (if not already connected)
                if not self.connection or not self.connection.is_connected():
                    if not self.connect_to_mysql():
                        return False
                
                # Step 3: Data cleaning
                logger.info("🧹 Step 2: Running data cleaning...")
                if not self.run_data_cleaning():
                    return False
                
                # Step 4: Feature engineering
                logger.info("🔧 Step 3: Running feature engineering...")
                if not self.run_feature_engineering():
                    return False
                
                # Step 5: Get processed data
                logger.info("📊 Step 4: Retrieving processed data...")
                processed_data = self.get_processed_data()
                if processed_data is None:
                    return False
                
                # Step 6: Upload to Azure (optional)
                if upload_to_azure:
                    logger.info("☁️ Step 5: Uploading results to Azure...")
                    self.upload_to_azure(processed_data)
                
                # Step 7: Generate report
                logger.info("📋 Step 6: Generating ETL report...")
                report = self.generate_etl_report()
                
                # Step 8: Save processed data locally for ML pipeline
                logger.info("💾 Step 7: Saving processed data locally...")
                processed_data.to_csv('processed_data_for_ml.csv', index=False)
                logger.info("💾 Saved processed data for ML pipeline")
                
                # Step 9: Save ETL report
                logger.info("📄 Step 8: Saving ETL report...")
                import json
                with open('etl_report.json', 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info("📄 Saved ETL report")
                
                pipeline_logger.info("🎉 Full ETL pipeline completed successfully!")
                return True
                
            except Exception as e:
                pipeline_logger.error(f"❌ ETL pipeline failed: {e}")
                return False
            finally:
                self.disconnect_from_mysql() 